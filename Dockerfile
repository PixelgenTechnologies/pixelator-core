FROM rust:1.90-trixie AS builder
WORKDIR /usr/src/pixelator-core
COPY ./packages ./packages
COPY ./Cargo.* .

RUN cargo install --path ./packages/cli

FROM debian:trixie-slim

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/cargo/bin/community-detection /usr/local/bin/community-detection
CMD ["community-detection"]
