FROM golang:1.19-alpine

WORKDIR /app

# Copia e baixa dependências
COPY go.mod ./
# Não temos dependências externas complexas, mas é boa prática
# RUN go mod download 

COPY main.go ./

# Compila o programa
RUN go build -o coordinator main.go

# Roda o programa
CMD ["./coordinator"]