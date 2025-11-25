// coordinator/GenerateCSV.go
package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Lista de empresas do S&P500 usadas no CSV
var sp500Tickers = []string{
	"AAPL", "MSFT", "AMZN", "NVDA", "META",
	"GOOGL", "GOOG", "TSLA", "JPM", "JNJ",
	"V", "PG", "XOM", "HD", "UNH",
	"MA", "BAC", "PEP", "COST", "PFE",
	"DIS", "KO", "CSCO", "T", "ABT",
	"INTC", "MRK", "WMT", "ORCL", "CVX",
	"NKE", "LLY", "MCD", "DHR", "ACN",
	"MDT", "AMGN", "BMY", "TXN", "NEE",
	"IBM", "HON", "AMD", "CAT", "GS",
	"GE", "UPS", "UNP", "QCOM", "ADBE",
	"AVGO", "CRM", "LIN", "ABBV", "ABNB",
	"BKNG", "SPGI", "BLK", "ADP", "ISRG",
	"ELV", "HCA", "TMO", "LOW", "MS",
	"AXP", "C", "DE", "RTX", "LMT",
	"PM", "MDLZ", "SBUX", "INTU", "AMAT",
	"NOW", "ADI", "LRCX", "MU", "PLD",
	"EQIX", "CCI", "CB", "PGR", "SO",
	"DUK", "REGN", "VRTX", "PANW", "COP",
	"OXY", "CSX", "NSC", "KMB", "CL",
	"MO", "BK", "USB", "EOG", "TGT",
}

// Busca histórico diário de um ticker no Stooq
func fetchStooqHistory(ticker string) (map[string]float64, error) {
	// Stooq usa sufixo .US para ações americanas
	symbol := strings.ToLower(ticker) + ".us"

	url := fmt.Sprintf(
		"https://stooq.com/q/d/l/?s=%s&i=d",
		symbol,
	)

	resp, err := http.Get(url)
	if err != nil {
		return nil, fmt.Errorf("erro ao buscar %s no Stooq: %w", ticker, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("status %d ao buscar %s: %s", resp.StatusCode, ticker, string(body))
	}

	reader := csv.NewReader(resp.Body)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("erro lendo CSV de %s: %w", ticker, err)
	}

	prices := make(map[string]float64)

	// Esperado: Date,Open,High,Low,Close,Volume
	for i, rec := range records {
		if i == 0 {
			// cabeçalho
			continue
		}
		if len(rec) < 5 {
			continue
		}

		date := rec[0]
		closeStr := rec[4]

		if closeStr == "" || closeStr == "null" {
			continue
		}

		v, err := strconv.ParseFloat(closeStr, 64)
		if err != nil {
			continue
		}
		prices[date] = v
	}

	if len(prices) == 0 {
		return nil, fmt.Errorf("stooq retornou 0 pontos para %s", ticker)
	}

	return prices, nil
}

// Gera o arquivo ../data/portfolio_allocation.csv com numAssets ativos
func PrepareHistoricalCSV(numAssets int) error {
	if numAssets <= 0 {
		return fmt.Errorf("numAssets deve ser > 0")
	}
	if numAssets > len(sp500Tickers) {
		numAssets = len(sp500Tickers)
	}

	// Embaralha a lista e pega os primeiros N
	rand.Seed(time.Now().UnixNano())
	shuffled := make([]string, len(sp500Tickers))
	copy(shuffled, sp500Tickers)
	rand.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})
	tickers := shuffled[:numAssets]

	log.Printf("Usando %d ativos do S&P500 (Stooq): %v\n", numAssets, tickers)

	allPrices := make(map[string]map[string]float64) // ticker -> (date -> price)
	var commonDates map[string]struct{}

	for i, t := range tickers {
		prices, err := fetchStooqHistory(t)
		if err != nil {
			return err
		}

		allPrices[t] = prices

		if i == 0 {
			// primeira vez: conjunto inicial de datas
			commonDates = make(map[string]struct{})
			for d := range prices {
				commonDates[d] = struct{}{}
			}
		} else {
			// interseção com datas já vistas
			for d := range commonDates {
				if _, ok := prices[d]; !ok {
					delete(commonDates, d)
				}
			}
		}

		// Pequena pausa de cortesia (não é obrigatório, mas é simpático)
		time.Sleep(200 * time.Millisecond)
	}

	if len(commonDates) == 0 {
		return fmt.Errorf("nenhuma data em comum entre os ativos escolhidos")
	}

	// Ordena datas
	dates := make([]string, 0, len(commonDates))
	for d := range commonDates {
		dates = append(dates, d)
	}
	sort.Strings(dates)

	// Pasta ../data em relação ao diretório do coordinator
	dataDir := filepath.Join("..", "data")

	// Se a pasta não existir, cria ela (isso evita falha na inicialização do coordenador)
	if _, err := os.Stat(dataDir); err != nil {
		if os.IsNotExist(err) {
			// cria o diretório data (incluindo pais, caso faltem)
			if mkerr := os.MkdirAll(dataDir, 0o755); mkerr != nil {
				return fmt.Errorf("erro criando diretório de dados em %s: %w", dataDir, mkerr)
			}
			log.Printf("Diretório de dados criado em %s", dataDir)
		} else {
			return fmt.Errorf("erro ao acessar diretório de dados %s: %w", dataDir, err)
		}
	}

	outPath := filepath.Join(dataDir, "portfolio_allocation.csv")
	f, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("erro criando CSV: %w", err)
	}
	defer f.Close()

	writer := csv.NewWriter(f)

	// Cabeçalho: date, T1, T2, ...
	header := make([]string, 0, len(tickers)+1)
	header = append(header, "date")
	header = append(header, tickers...)
	if err := writer.Write(header); err != nil {
		return fmt.Errorf("erro escrevendo header: %w", err)
	}

	// Linhas: date, price1, price2, ...
	for _, d := range dates {
		row := make([]string, 0, len(tickers)+1)
		row = append(row, d)
		for _, t := range tickers {
			price := allPrices[t][d]
			row = append(row, fmt.Sprintf("%.6f", price))
		}
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("erro escrevendo linha: %w", err)
		}
	}

	writer.Flush()
	if err := writer.Error(); err != nil {
		return fmt.Errorf("erro finalizando CSV: %w", err)
	}

	log.Printf("CSV gerado em %s com %d dias e %d ativos\n", outPath, len(dates), len(tickers))
	return nil
}
