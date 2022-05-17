FROM rust:1.60-buster

RUN apt-get update && apt-get install -y bash curl coreutils libc-dev libssl-dev openssl

COPY . /app

WORKDIR /app

RUN cargo build --release

ENTRYPOINT [ "/app/target/release/cellfie-runtime-predictor" ]
