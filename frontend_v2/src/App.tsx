import React, { useEffect, useRef, useState } from 'react';
import {
  LayoutDashboard, LineChart, Briefcase, History, Settings,
  TrendingUp, Sparkles, Bell, User, Search, MessageSquare, Send
} from 'lucide-react';
import { createChart, ColorType, CandlestickSeries } from 'lightweight-charts';

const App: React.FC = () => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const [activeTab, setActiveTab] = useState('Hàng hóa');
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [activeAsset, setActiveAsset] = useState('GOLD');
  const [sidebarView, setSidebarView] = useState<'market' | 'ai'>('market');
  const [currentPrice, setCurrentPrice] = useState(2154.30);
  const [priceChange, setPriceChange] = useState(-0.24);
  const [chatMessages, setChatMessages] = useState([
    { role: 'assistant', text: 'Chào bạn! Tôi là AI hỗ trợ giao dịch. Bạn cần giúp gì về thị trường Vàng hôm nay?' }
  ]);
  const [inputValue, setInputValue] = useState('');

  useEffect(() => {
    if (!chartContainerRef.current) return;

    let simulationInterval: any = null;

    try {
      const chart: any = createChart(chartContainerRef.current, {
        layout: {
          background: { type: ColorType.Solid, color: '#0A0E14' },
          textColor: '#8B949E' as any,
        },
        grid: {
          vertLines: { color: '#161B22' },
          horzLines: { color: '#161B22' },
        },
        width: chartContainerRef.current.clientWidth,
        height: chartContainerRef.current.clientHeight || 450,
      });

      const candlestickSeries = chart.addSeries(CandlestickSeries, {
        upColor: '#10B981',
        downColor: '#EF4444',
        borderVisible: false,
        wickUpColor: '#10B981',
        wickDownColor: '#EF4444',
      });

      const startSimulation = (lastPoint: any) => {
        let current = { ...lastPoint };
        let lastMinute = Math.floor(Date.now() / 60000);

        simulationInterval = setInterval(() => {
          const nowMinute = Math.floor(Date.now() / 60000);

          if (nowMinute > lastMinute) {
            // Start a new candle every minute
            const newTime = current.time + 60;
            current = {
              time: newTime as any,
              open: current.close,
              high: current.close,
              low: current.close,
              close: current.close,
            };
            lastMinute = nowMinute;
          } else {
            // Smooth, multi-tick update for the current candle
            // Fast volatility for testing: 0.2% of the current price per second
            const volatility = current.close * 0.002;
            const change = (Math.random() - 0.5) * volatility;
            current.close += change;
            if (current.close > current.high) current.high = current.close;
            if (current.close < current.low) current.low = current.close;
          }

          setCurrentPrice(current.close);
          candlestickSeries.update(current);
        }, 1000);
      };

      const fetchData = async () => {
        try {
          const response = await fetch('http://localhost:8000/api/market-data');
          const data = await response.json();
          if (Array.isArray(data) && data.length > 0) {
            candlestickSeries.setData(data);
            chart.timeScale().fitContent();

            const last = data[data.length - 1];
            setCurrentPrice(last.close);

            // Start live simulation based on last data point
            startSimulation(last);
          } else {
            const mockData = generateMockData();
            candlestickSeries.setData(mockData);
            startSimulation(mockData[mockData.length - 1]);
          }
        } catch (err) {
          const mockData = generateMockData();
          candlestickSeries.setData(mockData);
          startSimulation(mockData[mockData.length - 1]);
        }
      };

      fetchData();

      const handleResize = () => {
        if (chartContainerRef.current) {
          chart.applyOptions({
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight
          });
        }
      };

      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
        chart.remove();
        if (simulationInterval) {
          clearInterval(simulationInterval);
        }
      };
    } catch (err) {
      console.error("Chart init error:", err);
    }
  }, []);

  const generateMockData = (): any[] => {
    const data: any[] = [];
    let currentPrice = 1.15181;
    let time = Math.floor(Date.now() / 1000) - 100 * 3600;

    for (let i = 0; i < 100; i++) {
      const open = currentPrice + (Math.random() - 0.5) * 0.01;
      const high = open + Math.random() * 0.005;
      const low = open - Math.random() * 0.005;
      const close = low + Math.random() * (high - low);
      data.push({ time: time as any, open, high, low, close });
      currentPrice = close;
      time += 3600;
    }
    return data;
  };

  const handleSendMessage = () => {
    if (!inputValue.trim()) return;
    const newMessages = [...chatMessages, { role: 'user', text: inputValue }];
    setChatMessages(newMessages);
    setInputValue('');

    // Simple mock response
    setTimeout(() => {
      setChatMessages([...newMessages, { role: 'assistant', text: 'Tôi đang phân tích dữ liệu thị trường cho bạn...' }]);
    }, 800);
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '60px 280px 1fr 300px', height: '100vh', width: '100vw' }}>
      {/* Sidebar */}
      <nav style={styles.sidebar}>
        <div style={styles.logo}>TT</div>
        <NavItem active={sidebarView === 'market'} title="Thị trường" icon={<LayoutDashboard size={20} />} onClick={() => setSidebarView('market')} />
        <NavItem title="Biểu đồ" icon={<LineChart size={20} />} />
        <NavItem title="Danh mục" icon={<Briefcase size={20} />} />
        <NavItem title="Lịch sử" icon={<History size={20} />} />
        <NavItem active={sidebarView === 'ai'} title="AI support" icon={<MessageSquare size={20} />} onClick={() => setSidebarView('ai')} />

        <div style={{ marginTop: 'auto' }}>
          <NavItem title="Cài đặt" icon={<Settings size={20} />} />
        </div>
      </nav>

      {/* Second Column: Market or AI Chat */}
      <section style={styles.marketSidebar}>
        {sidebarView === 'market' ? (
          <>
            <div style={styles.marketHeader}>Thị trường</div>
            <div style={styles.marketTabs}>
              {['Hàng hóa', 'Tiền tệ', 'Tiền điện tử'].map(tab => (
                <span
                  key={tab}
                  style={{ ...styles.marketTab, ...(activeTab === tab ? styles.marketTabActive : {}) }}
                  onClick={() => setActiveTab(tab)}
                >
                  {tab}
                </span>
              ))}
            </div>
            <div style={styles.searchBar}>
              <div style={styles.inputWrapper}>
                <Search size={14} color="#8B949E" style={{ marginRight: '8px' }} />
                <input type="text" placeholder="Tìm kiếm tài sản..." style={styles.ghostInput} />
              </div>
            </div>
            <div style={styles.marketItems}>
              <MarketItem
                symbol="GOLD"
                name="XAUUSD"
                price={currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                change={priceChange > 0 ? `+${priceChange}%` : `${priceChange}%`}
                active={activeAsset === 'GOLD'}
                onClick={() => setActiveAsset('GOLD')}
              />
              {/* Other assets removed as per request */}
            </div>
          </>
        ) : (
          <div style={styles.chatContainer}>
            <div style={styles.marketHeader}>AI Support</div>
            <div style={styles.chatMessages}>
              {chatMessages.map((msg, i) => (
                <div key={i} style={{
                  ...styles.message,
                  alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                  background: msg.role === 'user' ? 'rgba(16, 185, 129, 0.2)' : '#0D1117'
                }}>
                  {msg.text}
                </div>
              ))}
            </div>
            <div style={styles.chatInput}>
              <input
                type="text"
                placeholder="Hỏi AI..."
                style={styles.ghostInput}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
              />
              <Send size={16} style={{ cursor: 'pointer', color: '#10B981' }} onClick={handleSendMessage} />
            </div>
          </div>
        )}
      </section>

      {/* Main Content */}
      <main style={styles.mainContent}>
        <header style={styles.topNav}>
          <div style={styles.pairInfo}>
            <TrendingUp color="#10B981" size={20} />
            <h2 style={{ fontSize: '1.1em', fontWeight: 600 }}>
              {activeAsset} <span style={{ fontSize: '0.6em', color: '#8B949E', fontWeight: 400 }}>Real-time</span>
            </h2>
            <button style={styles.btnAi} onClick={() => setSidebarView('ai')}>
              <Sparkles size={14} /> Hỏi AI
            </button>
          </div>
          <div style={styles.accountInfo}>
            <div style={styles.balanceBox}>
              <span style={{ fontSize: '0.7em', color: '#8B949E', display: 'block' }}>Demo Balance</span>
              <strong style={{ fontSize: '0.9em' }}>$25,182.45</strong>
            </div>
            <div style={styles.iconBtn}><Bell size={18} /></div>
            <div style={styles.iconBtn}><User size={18} /></div>
          </div>
        </header>

        <div ref={chartContainerRef} style={{ flex: 1, position: 'relative', background: '#0A0E14' }}>
          {/* Chart container */}
        </div>
      </main>

      {/* Order Panel */}
      <section style={styles.orderPanel}>
        <div style={{ fontSize: '1em', fontWeight: 600 }}>Đặt lệnh nhanh</div>
        <div style={styles.buySellToggle}>
          <button
            style={{ ...styles.toggleBtn, ...(side === 'sell' ? styles.toggleBtnSell : {}) }}
            onClick={() => setSide('sell')}
          >
            BÁN
          </button>
          <button
            style={{ ...styles.toggleBtn, ...(side === 'buy' ? styles.toggleBtnBuy : {}) }}
            onClick={() => setSide('buy')}
          >
            MUA
          </button>
        </div>

        <div style={styles.inputGroup}>
          <label style={styles.label}>Số lượng</label>
          <div style={styles.inputBox}>
            <input type="number" defaultValue="1" style={styles.ghostInput} />
          </div>
        </div>

        <div style={{ marginTop: 'auto', textAlign: 'center' }}>
          <button style={{
            ...styles.actionBtn,
            background: side === 'buy' ? '#10B981' : '#EF4444',
            color: side === 'buy' ? 'black' : 'white',
            boxShadow: `0 8px 30px ${side === 'buy' ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.4)'}`
          }}>
            Đặt Lệnh {side === 'buy' ? 'Mua' : 'Bán'}
          </button>
          <div style={styles.footerBranding}>
            Designed with 💚 by Nguyen Thien Thanh
          </div>
        </div>
      </section>
    </div>
  );
};

const NavItem: React.FC<{ icon: React.ReactNode, active?: boolean, title?: string, onClick?: () => void }> = ({ icon, active, title, onClick }) => (
  <div
    onClick={onClick}
    title={title}
    style={{
      ...styles.navItem,
      ...(active ? { color: '#10B981', background: 'rgba(16, 185, 129, 0.1)' } : {})
    }}
  >
    {icon}
  </div>
);

const MarketItem: React.FC<{ symbol: string, name: string, price: string, change: string, up?: boolean, down?: boolean, active?: boolean, onClick: () => void }> = ({ symbol, name, price, change, up, down, active, onClick }) => (
  <div onClick={onClick} style={{ ...styles.marketItem, ...(active ? { background: '#1C2128' } : {}) }}>
    <div>
      <span style={{ fontWeight: 600, display: 'block', fontSize: '0.9em' }}>{symbol}</span>
      <span style={{ fontSize: '0.7em', color: '#8B949E' }}>{name}</span>
    </div>
    <div style={{ textAlign: 'right' }}>
      <span style={{ fontWeight: 600, display: 'block', fontSize: '0.9em' }}>{price}</span>
      <span style={{ fontSize: '0.7em', color: up ? '#10B981' : (down ? '#EF4444' : '#8B949E') }}>{change}</span>
    </div>
  </div>
);

const styles: Record<string, React.CSSProperties> = {
  sidebar: { background: '#161B22', borderRight: '1px solid #30363D', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '20px 0', gap: '20px' },
  logo: { width: '32px', height: '32px', background: '#10B981', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'black', fontWeight: 'bold', marginBottom: '10px' },
  navItem: { color: '#8B949E', cursor: 'pointer', padding: '10px', borderRadius: '10px', transition: '0.2s' },
  marketSidebar: { background: '#161B22', borderRight: '1px solid #30363D', display: 'flex', flexDirection: 'column' },
  marketHeader: { padding: '20px', fontSize: '1em', fontWeight: 600 },
  marketTabs: { display: 'flex', padding: '0 20px 10px', gap: '15px', borderBottom: '1px solid #30363D' },
  marketTab: { fontSize: '0.8em', color: '#8B949E', cursor: 'pointer', paddingBottom: '5px' },
  marketTabActive: { color: '#E6EDF3', borderBottom: '2px solid #10B981' },
  searchBar: { padding: '15px 20px' },
  inputWrapper: { background: '#0D1117', border: '1px solid #30363D', padding: '8px 12px', borderRadius: '8px', display: 'flex', alignItems: 'center' },
  ghostInput: { background: 'transparent', border: 'none', color: 'white', outline: 'none', width: '100%', fontSize: '0.9em' },
  marketItems: { flex: 1, overflowY: 'auto' },
  marketItem: { display: 'flex', justifyContent: 'space-between', padding: '12px 20px', cursor: 'pointer', borderBottom: '1px solid rgba(48, 54, 61, 0.3)' },
  mainContent: { display: 'flex', flexDirection: 'column' },
  topNav: { height: '60px', background: '#161B22', borderBottom: '1px solid #30363D', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 20px' },
  pairInfo: { display: 'flex', alignItems: 'center', gap: '12px' },
  btnAi: { background: 'linear-gradient(135deg, #10B981 0%, #059669 100%)', border: 'none', padding: '5px 12px', borderRadius: '20px', color: 'white', fontSize: '0.75em', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', boxShadow: '0 4px 15px rgba(16, 185, 129, 0.2)' },
  accountInfo: { display: 'flex', alignItems: 'center', gap: '15px' },
  balanceBox: { background: 'rgba(255, 255, 255, 0.05)', padding: '5px 12px', borderRadius: '8px' },
  iconBtn: { color: '#8B949E', cursor: 'pointer' },
  orderPanel: { background: '#161B22', borderLeft: '1px solid #30363D', padding: '20px', display: 'flex', flexDirection: 'column', gap: '15px' },
  buySellToggle: { display: 'grid', gridTemplateColumns: '1fr 1fr', background: '#0D1117', padding: '4px', borderRadius: '10px' },
  toggleBtn: { background: 'transparent', border: 'none', color: '#8B949E', padding: '10px', borderRadius: '8px', fontWeight: 600, cursor: 'pointer' },
  toggleBtnBuy: { background: '#10B981', color: 'black' },
  toggleBtnSell: { background: '#EF4444', color: 'white' },
  inputGroup: { display: 'flex', flexDirection: 'column', gap: '6px' },
  label: { fontSize: '0.7em', color: '#8B949E', textTransform: 'uppercase' },
  inputBox: { display: 'flex', alignItems: 'center', background: '#0D1117', border: '1px solid #30363D', padding: '10px 12px', borderRadius: '8px' },
  actionBtn: { width: '100%', padding: '14px', borderRadius: '12px', border: 'none', fontSize: '0.95em', fontWeight: 700, cursor: 'pointer', transition: '0.2s' },
  footerBranding: { marginTop: '15px', fontSize: '0.65em', color: '#8B949E' },
  chatContainer: { display: 'flex', flexDirection: 'column', height: '100%' },
  chatMessages: { flex: 1, padding: '20px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px' },
  message: { padding: '10px 14px', borderRadius: '12px', fontSize: '0.85em', maxWidth: '85%' },
  chatInput: { padding: '20px', borderTop: '1px solid #30363D', display: 'flex', gap: '10px', alignItems: 'center' }
};

export default App;
