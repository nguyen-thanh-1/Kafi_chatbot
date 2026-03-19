// Trading Dashboard Logic
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    setupInteractions();
});

function initChart() {
    const chartElement = document.getElementById('tv-chart');
    const chart = LightweightCharts.createChart(chartElement, {
        layout: {
            background: { type: 'solid', color: '#0A0E14' },
            textColor: '#8B949E',
        },
        grid: {
            vertLines: { color: '#161B22' },
            horzLines: { color: '#161B22' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
        },
        rightPriceScale: {
            borderColor: '#30363D',
        },
        timeScale: {
            borderColor: '#30363D',
            timeVisible: true,
            secondsVisible: false,
        },
    });

    const candlestickSeries = chart.addCandlestickSeries({
        upColor: '#10B981',
        downColor: '#EF4444',
        borderVisible: false,
        wickUpColor: '#10B981',
        wickDownColor: '#EF4444',
    });

    // Mock data for initial view
    const data = generateMockData();
    candlestickSeries.setData(data);

    // Responsive chart
    window.addEventListener('resize', () => {
        chart.applyOptions({
            width: chartElement.clientWidth,
            height: chartElement.clientHeight
        });
    });
}

function generateMockData() {
    const data = [];
    let currentPrice = 1.15181;
    let time = Math.floor(Date.now() / 1000) - 100 * 3600;

    for (let i = 0; i < 100; i++) {
        const open = currentPrice + (Math.random() - 0.5) * 0.001;
        const high = open + Math.random() * 0.0005;
        const low = open - Math.random() * 0.0005;
        const close = low + Math.random() * (high - low);

        data.push({
            time: time,
            open: open,
            high: high,
            low: low,
            close: close,
        });

        currentPrice = close;
        time += 3600;
    }
    return data;
}

function setupInteractions() {
    const buyBtn = document.querySelector('.toggle-btn.buy');
    const sellBtn = document.querySelector('.toggle-btn.sell');
    const actionBtn = document.querySelector('.action-btn');

    buyBtn.addEventListener('click', () => {
        buyBtn.classList.add('active');
        sellBtn.classList.remove('active');
        actionBtn.className = 'action-btn buy';
        actionBtn.textContent = 'Đặt Lệnh Mua';
    });

    sellBtn.addEventListener('click', () => {
        sellBtn.classList.add('active');
        buyBtn.classList.remove('active');
        actionBtn.className = 'action-btn sell';
        actionBtn.style.background = 'var(--accent-sell)';
        actionBtn.style.color = 'white';
        actionBtn.style.boxShadow = '0 5px 20px rgba(239, 68, 68, 0.4)';
        actionBtn.textContent = 'Đặt Lệnh Bán';
    });

    // Handle Market Item selection
    const marketItems = document.querySelectorAll('.market-item');
    marketItems.forEach(item => {
        item.addEventListener('click', () => {
            marketItems.forEach(i => i.style.background = 'transparent');
            item.style.background = 'var(--surface-hover)';
            const symbol = item.querySelector('.symbol').textContent;
            document.querySelector('.pair-info h2').firstChild.textContent = symbol + ' ';
        });
    });
}
