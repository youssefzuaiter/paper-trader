# Kubernetes deployment strategy: the paper-trader swarm

Four agent classes, one message bus, one gatekeeper. The gatekeeper is the
only pod that can place an order.

```
                 Redis Streams (StatefulSet)
Alpaca news WS ─▶ Ingestion ──news.raw──▶ Quantitative ──news.enriched──▶ Inference ──▶ POST /v1/signals (HMAC)
   (1 pod)          (1 pod)              (N pods, 1 group)               (N pods, HPA)         │
                                                                                               ▼
                                                                      Risk & Routing agent (StatefulSet, exactly 1)
                                                                        guard → limits → size → async submit ──▶ Alpaca paper
                                                                        exit monitor (stop / take-profit)
                                                                        signed receipts ──▶ PFW (outside the cluster)
```

## Workload choices

| Agent | Kind | Replicas | Why this kind | State | Scales by |
|---|---|---|---|---|---|
| **Ingestion** | Deployment, `Recreate` | 1 | Holds Alpaca's news WebSocket. The free data plan limits concurrent stream connections, and `Recreate` never runs two pods during a rollout. | None (dedup keys live in Redis) | Can't: one stream connection. Add symbols to the watchlist instead. |
| **Quantitative** | Deployment | 2+ | Pure function of news + bars. Replicas share one consumer group, so each item is processed once. | None (bar cache in Redis) | Replicas |
| **Inference** | Deployment + HPA | 2 to 4 | CPU-bound model serving (FinBERT int8 + scikit-learn). Models are baked into the image tag, so rollback is `kubectl rollout undo`. | None | CPU (HPA). KEDA on stream lag is the free upgrade. |
| **Risk & Routing** | **StatefulSet** | **exactly 1** | Must be a single writer: check-then-submit is atomic only inside one process. A StatefulSet stops ordinal 0 before starting its replacement, whereas a Deployment's rolling update overlaps pods. It also needs a PVC for the kill switch, breaker latch and receipt outbox. | PVC 256 Mi | Doesn't. It makes a few decisions per minute. |
| **Redis** (bus) | StatefulSet | 1 | AOF persistence on a PVC. `noeviction`, so a full bus refuses writes rather than dropping news. | PVC 1 Gi | Vertically |

**Why REST for signals and a queue for news:** news fans out and should
buffer through restarts, which is the queue's job. A signal needs an answer
(accepted, rejected, or 423 "stop sending"), and the gatekeeper wants
backpressure rather than a backlog of stale signals. Stale ones are
rejected after 120 s regardless.

## Least-privilege secrets

| Secret | Ingestion | Quant | Inference | Router |
|---|:-:|:-:|:-:|:-:|
| `alpaca-paper` (paper keys) | ✓ | ✓ | – | ✓ |
| `router-signal-hmac` (inference → router) | – | – | ✓ | ✓ |
| `pfw-webhook` (router ↔ PFW) | – | – | – | ✓ |

A compromised inference pod can submit signals, which still pass every
limit, but it cannot forge receipts to PFW, trigger a halt, or call Alpaca.
A NetworkPolicy admits only `trader.io/role: inference` pods (plus the
ingress controller, for PFW's control calls) to the router. Every pod runs
non-root, with a read-only root filesystem, all capabilities dropped, and
under the namespace's `restricted` Pod Security level.

## Where to run it

The whole stack is free to run if you host it yourself:

- **Docker Desktop Kubernetes** (already on this machine): what the smoke
  test below used.
- **k3s** on a spare machine, or on a free-tier cloud VM such as Oracle
  Cloud's Always Free ARM instances (check their current terms).

Paid options, flagged so they're a choice rather than a surprise: managed
Kubernetes (GKE, EKS, DigitalOcean) bills for nodes and usually for the
control plane or load balancer too. Render, where the monolith runs today,
does not run Kubernetes. A 512 MB Render instance can't host this stack.

## Deploy

```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl -n paper-trader create secret generic alpaca-paper \
  --from-literal=api-key-id=... --from-literal=api-secret-key=...
kubectl -n paper-trader create secret generic router-signal-hmac \
  --from-literal=secret="$(python3 -c 'import secrets;print(secrets.token_hex(32))')"
kubectl -n paper-trader create secret generic pfw-webhook --from-literal=secret=<same WEBHOOK_SECRET as PFW>
docker build -f Dockerfile.router    -t paper-trader/risk-router:0.1.0 .
docker build -f Dockerfile.swarm     -t paper-trader/swarm:0.1.0 .      # Ingestion + Quant
docker build -f Dockerfile.inference -t paper-trader/inference:0.1.0 .  # needs models/return_model/
kubectl apply -k deploy/k8s          # everything
```

Docker Desktop's Kubernetes uses local images directly. For another
cluster, push the three images to a registry it can pull from (GitHub
Container Registry is free for public images).

## Operating the gatekeeper

- **Never scale `risk-router` above 1.** Two replicas can both pass the
  exposure check and double-buy.
- **Halt:** a signed `POST /v1/control/halt` (PFW's existing halt button,
  pointed at the router) cancels working orders and blocks every order,
  exits included. It is stored on the PVC, so pod restarts don't clear it.
  **Resume** is deliberately manual:
  `kubectl -n paper-trader exec risk-router-0 -- rm /app/state/router-state.json`
  then `kubectl -n paper-trader delete pod risk-router-0`.
- **Circuit breaker:** at −2.5% daily P&L, new positions are refused
  (HTTP 423) until the next New York trading day, even if equity recovers
  or the pod restarts. Exits keep running. If P&L can't be read, new
  positions are refused too (fail closed).
- `GET /health` reports the halt state, the breaker latch, whether the
  exit loop is running, the pending outbox count and every limit.

## Migrating off the monolith

1. **Deploy the swarm** (`kubectl apply -k deploy/k8s`).
2. **Set `ORDERS_VIA_RISK_ROUTER=true` on the monolith** (Render →
   Environment). It then places no orders at all: `/signals/execute`
   answers 410 and the autonomous loop does not start. It keeps its
   settlement stream, outbox and reconciler, which settle every fill on the
   account, including the router's buys and exits, so PFW books them with
   no changes on its side. Until this is set, the monolith can still buy
   past the router's portfolio limits.
3. **Point PFW's halt button at the router** (`POST /v1/control/halt`,
   same signature scheme), or keep both: halting the monolith no longer
   matters once step 2 is done.
4. **Later: fold settlement into the router** (the trade_updates stream
   plus the hourly reconciler) and retire the monolith.

## Retraining the return model

```bash
./.venv/bin/python -m swarm.train_return_model --start 2024-01-02 --end <a week ago>
docker build -f Dockerfile.inference -t paper-trader/inference:<new tag> .
```

Read `models/return_model/report.md` before shipping a new version. The
model is baked into the image, so rolling back is
`kubectl -n paper-trader rollout undo deployment/inference`.

## Verified on 2026-09-25 (Docker Desktop Kubernetes v1.34.1)

**Router alone, dummy Alpaca keys.** `kubectl apply -k` admitted it under
`restricted` Pod Security.

| Check | Result |
|---|---|
| Unsigned signal | 403 |
| Signed signal, Alpaca unreachable (bad keys) | 423 `breaker_unavailable` (fails closed) |
| Signed halt | 200, `halted: true` |
| `kubectl delete pod risk-router-0`, then `/health` | `halted: true` (read back from the PVC) |
| Signed signal after the restart | 423 `halted` |

**Whole swarm, during market hours.** Ingestion and Quant used the real
paper keys (read-only market data); the router had dummy keys, so it could
not trade.

| Check | Result |
|---|---|
| All five workloads | Running under `restricted` Pod Security. Quant first crash-looped on a file missing from `Dockerfile.swarm`; `tests/test_images.py` now catches that class of bug. |
| Ingestion | Connected to the live news stream and backfilled 6 h (27 articles). A restarted pod published 0: it resumed from the newest article already sent. |
| Quant → Inference | 46 article-symbol items enriched and evaluated, 0 failures, 0 dead letters |
| Model on live news | prob_up 0.358–0.450, the same range as the held-out test data (0.33–0.48), so live features match training. All below the router's 0.60, so nothing was sent. |

The namespace, including the Secret holding the real keys, was deleted afterwards.
