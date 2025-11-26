# syntax=docker/dockerfile:1

FROM --platform=$BUILDPLATFORM golang:1.24 AS builder
WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Ensure go.mod and go.sum reflect current imports
RUN go mod tidy

# Let Buildx set GOOS/GOARCH automatically
RUN CGO_ENABLED=0 go build -o /out/mutator ./cmd/mutator
RUN CGO_ENABLED=0 go build -o /out/taintcontroller ./cmd/taintcontroller
RUN CGO_ENABLED=0 go build -o /out/metrics ./cmd/metrics

FROM --platform=$TARGETPLATFORM debian:bookworm-slim

# Install Python + pip + dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-requests python3-pip python3-kubernetes ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Python libraries


WORKDIR /app

COPY cmd/poll ./cmd/poll
COPY poller ./poller
COPY --from=builder /out /app/bin
COPY entrypoint.sh /usr/local/bin/carbon-kube

RUN chmod +x /usr/local/bin/carbon-kube

ENV PYTHONPATH="/app"
ENTRYPOINT ["carbon-kube"]