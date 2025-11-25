FROM golang:1.19-alpine

WORKDIR /app
COPY go.mod ./
COPY main.go GenerateCSV.go ./
RUN go build -o coordinator main.go GenerateCSV.go

CMD ["./coordinator"]