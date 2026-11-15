FROM node:24-alpine AS builder

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./frontend/
WORKDIR /app/frontend
RUN npm ci

WORKDIR /app
COPY frontend/ ./frontend/
COPY corfu_dataset.json ./corfu_dataset.json
WORKDIR /app/frontend
ENV VITE_API_BASE_URL=/api
RUN npm run build

FROM nginx:stable-alpine AS runtime

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/frontend/dist/ /usr/share/nginx/html/

EXPOSE 80
