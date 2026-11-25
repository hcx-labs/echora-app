FROM node:22.22.0-bookworm-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
COPY web/package.json web/package.json
RUN npm ci

FROM node:22.22.0-bookworm-slim AS builder
WORKDIR /app
ENV BACKEND_URL=http://backend:8000
COPY --from=deps /app/node_modules ./node_modules
COPY --from=deps /app/web/node_modules ./web/node_modules
COPY . .
RUN npm run build

FROM node:22.22.0-bookworm-slim AS runner
ENV NODE_ENV=production
WORKDIR /app
RUN useradd --system --uid 1001 echora
COPY --from=builder --chown=echora /app/web/.next/standalone ./
COPY --from=builder --chown=echora /app/web/.next/static ./web/.next/static
COPY --from=builder --chown=echora /app/web/public ./web/public
USER echora
EXPOSE 3000
CMD ["node", "web/server.js"]
