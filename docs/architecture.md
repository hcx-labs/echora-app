# Architecture

Echora separates immutable track identity, recording equivalence, model representations, and user visibility.

## Services

The Next.js web service renders the interface; the browser calls the backend's API at `/api/v1/*` (nginx routes it to the backend directly, `next dev` through a rewrite; routes are listed in [the v1 route map](design/api-v1-routes.md)). The root `PlayerProvider` keeps playback and queues alive across routes.

The FastAPI backend (`backend/`, image `echora-backend`) owns OIDC sessions, application APIs, playback proxying, interactive inference, clustering, and job submission. Background work runs outside the API through PostgreSQL-backed jobs. Analysis workers process sync, import, and backfill batches; scheduled workers enqueue due curations and execute both automatic and manual refreshes. Claims, leases, retries, cancellation, and owner-scoped history are durable. See [worker operations](jobs.md) and [the batch-worker decision](adr/0006-use-durable-jobs-with-batch-analysis-workers.md).

PostgreSQL 17 with pgvector is the canonical store. SQLAlchemy handles ordinary lifecycle queries. Reviewed PostgreSQL SQL remains in analytical paths where pgvector operations, CTEs, bulk reconciliation, or query plans need direct control. Alembic creates the v1 schema and applies every later revision during analysis startup.

## Identity and access

OIDC is the only login method. The normalized email claim is the immutable username and the provider subject binds the identity. Echora stores authorization, provisioning, blocking, preferences, and sessions locally. Credentials remain deployment-owned or encrypted at rest.

A track is identified by SHA-256 of its source bytes and a UUIDv5 derived from that digest. `user_track_links` determines which shared tracks each user can access. Synchronization removes stale links without deleting reusable metadata, fingerprints, or representations.

Chromaprint supplies evidence for non-destructive recording groups. A recording group never replaces canonical track identity.

## Representations

Echora stores MuQ-MuLan semantic audio, MERT acoustic audio, and BGE-M3 lyrics representations separately. Similarity blends combine normalized scores at query time. Missing lyrics have explicit behavior and never silently change the requested weights.

Readers select an exact model revision, configuration hash, and dimension through the current-representation views. Committed per-track artifacts remain visible while other tracks are processing. Separate attempt records retain audio and lyrics embedding failures. See [ranking and representation contracts](analysis-correctness.md).

Measured descriptors remain separate from embeddings. Curation sound profiles use library-relative pace, energy, brightness, motion, vocal-presence, and dynamics targets as a soft ranking signal. Key and beat data do not affect curation scoring. See [audio descriptors](audio-descriptors.md).

Pinned Hugging Face snapshots live outside the image. Production downloads missing snapshots into a persistent volume before analysis starts. The running service uses offline mode so a repository update cannot change inference without a configured revision change.

Communities are reproducible SNN-Leiden partitions of a fixed corpus. Concepts are overlapping textual associations and remain distinct from communities. Projection coordinates support presentation only and never define similarity.

## Curations

A curation is a durable recipe for a fully managed Navidrome playlist. Revisions preserve evidence, membership, ordering, achieved familiarity mix, and shuffle seed. Last.fm history determines recent familiarity and recurring local-time evidence. User timezone controls time-of-day assignment.

## Deployment

Production runs separate web (`echora-web`) and backend (`echora-backend`) deployments behind one ingress, plus analysis and scheduled worker deployments (`echora-worker`, `python -m worker.executor analysis|scheduled`) without public ports. The ingress routes `/api/v1/` to the backend. PostgreSQL and model snapshots use persistent storage. The model init container has network access; the backend and worker containers do not download model files at runtime.
