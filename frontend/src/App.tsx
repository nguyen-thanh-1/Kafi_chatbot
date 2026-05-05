import React, { useEffect, useRef, useState } from 'react';
import {
  LayoutDashboard, LineChart, Briefcase, History, Settings,
  TrendingUp, Sparkles, Bell, User, Search, MessageSquare, Send,
  Database, Trash2, FileText, CheckCircle2, AlertCircle, Loader2, Upload
} from 'lucide-react';
import { createChart, ColorType, CandlestickSeries } from 'lightweight-charts';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';



type MarkdownTagProps<T> = T & { children?: React.ReactNode };
type CodeProps = React.HTMLAttributes<HTMLElement> & { inline?: boolean; className?: string; children?: React.ReactNode };

const markdownComponents = {
  p: (props: MarkdownTagProps<React.HTMLAttributes<HTMLParagraphElement>>) => <p style={{ margin: '0.4em 0' }} {...props} />,
  ul: (props: MarkdownTagProps<React.HTMLAttributes<HTMLUListElement>>) => <ul style={{ margin: '0.4em 0', paddingLeft: '1.2em' }} {...props} />,
  ol: (props: MarkdownTagProps<React.HTMLAttributes<HTMLOListElement>>) => <ol style={{ margin: '0.4em 0', paddingLeft: '1.2em' }} {...props} />,
  li: (props: MarkdownTagProps<React.LiHTMLAttributes<HTMLLIElement>>) => <li style={{ margin: '0.15em 0' }} {...props} />,
  a: (props: MarkdownTagProps<React.AnchorHTMLAttributes<HTMLAnchorElement>>) => (
    <a
      {...props}
      target="_blank"
      rel="noreferrer noopener"
      style={{ color: '#10B981', textDecoration: 'underline' }}
    />
  ),
  code: ({ inline, children, ...props }: CodeProps) => {
    if (inline) {
      return (
        <code
          {...props}
          style={{
            background: 'rgba(255,255,255,0.06)',
            border: '1px solid rgba(48,54,61,0.6)',
            padding: '0 0.35em',
            borderRadius: 6,
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
            fontSize: '0.92em',
          }}
        >
          {children}
        </code>
      );
    }

    return (
      <pre
        style={{
          margin: '0.6em 0',
          padding: '0.8em 0.9em',
          background: '#0D1117',
          border: '1px solid #30363D',
          borderRadius: 10,
          overflowX: 'auto',
        }}
      >
        <code
          {...props}
          style={{
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
            fontSize: '0.9em',
            lineHeight: 1.55,
          }}
        >
          {children}
        </code>
      </pre>
    );
  },
  table: (props: MarkdownTagProps<React.TableHTMLAttributes<HTMLTableElement>>) => (
    <div style={{ overflowX: 'auto', margin: '0.6em 0' }}>
      <table
        {...props}
        style={{
          width: '100%',
          borderCollapse: 'collapse',
          border: '1px solid #30363D',
          borderRadius: 10,
          overflow: 'hidden',
        }}
      />
    </div>
  ),
  th: (props: MarkdownTagProps<React.ThHTMLAttributes<HTMLTableCellElement>>) => (
    <th
      {...props}
      style={{
        textAlign: 'left',
        padding: '8px 10px',
        borderBottom: '1px solid #30363D',
        background: 'rgba(255,255,255,0.04)',
        fontWeight: 700,
      }}
    />
  ),
  td: (props: MarkdownTagProps<React.TdHTMLAttributes<HTMLTableCellElement>>) => (
    <td
      {...props}
      style={{
        padding: '8px 10px',
        borderBottom: '1px solid rgba(48,54,61,0.6)',
        verticalAlign: 'top',
      }}
    />
  ),
} as const;

const splitPipeRow = (line: string) => {
  const t = line.trim();
  if (!t.includes('|')) return null;

  // Support both `| a | b |` and `a | b` styles.
  const rawParts = t.split('|').map(s => s.trim());
  const hasEdgePipes = t.startsWith('|') && t.endsWith('|');
  const cells = hasEdgePipes ? rawParts.slice(1, -1) : rawParts;
  if (cells.length < 2) return null;
  return { cells, hasEdgePipes };
};

const isSeparatorCells = (cells: string[]) =>
  cells.length >= 2 && cells.every(c => /^:?-{3,}:?$/.test(c.trim()));

const toPipeLine = (cells: string[]) => `| ${cells.map(c => c.trim()).join(' | ')} |`;

const normalizeMarkdownTables = (input: string) => {
  const lines = input.split(/\r?\n/);
  const out: string[] = [];

  const isTableRowLine = (line: string) => {
    const row = splitPipeRow(line);
    return !!row && !isSeparatorCells(row.cells);
  };

  const isSeparatorLine = (line: string) => {
    const row = splitPipeRow(line);
    return !!row && isSeparatorCells(row.cells);
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (!isTableRowLine(line)) {
      out.push(line);
      continue;
    }

    const rows: string[][] = [];
    let hasSeparator = false;
    let separatorIndex = -1;

    let j = i;
    while (j < lines.length) {
      const l = lines[j];
      if (isTableRowLine(l)) {
        const parsed = splitPipeRow(l);
        if (parsed) rows.push(parsed.cells);
        j++;
        continue;
      }
      if (isSeparatorLine(l)) {
        hasSeparator = true;
        if (separatorIndex === -1) separatorIndex = rows.length;
        j++;
        continue;
      }
      break;
    }

    // If there was no explicit separator, add one.
    // - 1 row: treat it as body row, add an empty header.
    // - >=2 rows: treat first row as header.
    if (!hasSeparator) {
      const cols = rows[0]?.length ?? 0;
      if (cols >= 2) {
        if (rows.length === 1) {
          out.push(toPipeLine(Array.from({ length: cols }, () => '')));
          out.push(toPipeLine(Array.from({ length: cols }, () => '---')));
          out.push(toPipeLine(rows[0]));
        } else {
          out.push(toPipeLine(rows[0]));
          out.push(toPipeLine(Array.from({ length: cols }, () => '---')));
          for (const r of rows.slice(1)) out.push(toPipeLine(r));
        }
      } else {
        out.push(line);
      }
      i = j - 1;
      continue;
    }

    // With a separator present, normalize to GFM table format and drop any "separator-like" body rows
    // (common when the model outputs both a separator row and an extra `-----` row).
    const header = rows[0] ?? [];
    const cols = header.length;
    out.push(toPipeLine(header));
    out.push(toPipeLine(Array.from({ length: cols }, () => '---')));

    // Body rows start at index 1 in `rows` when the markdown is well-formed.
    // If the model inserted extra separator-like rows, remove them.
    for (const r of rows.slice(1)) {
      if (isSeparatorCells(r)) continue;
      out.push(toPipeLine(r));
    }

    i = j - 1;
  }

  return out.join('\n');
};

const App: React.FC = () => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chatMessagesRef = useRef<HTMLDivElement>(null);
  const apiUrl =
    import.meta.env.VITE_API_URL ||
    '';
  const [activeTab, setActiveTab] = useState('Hàng hóa');
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [activeAsset, setActiveAsset] = useState('GOLD');
  const [sidebarView, setSidebarView] = useState<'market' | 'ai' | 'knowledge'>('market');
  const NAV_WIDTH = 60;
  const MARKET_WIDTH = 280;
  const ORDER_MIN_WIDTH = 220;
  const ORDER_MAX_WIDTH = 300;
  const CHAT_MIN_WIDTH = 280;
  const [chatWidth, setChatWidth] = useState(CHAT_MIN_WIDTH);
  const [isChatResizing, setIsChatResizing] = useState(false);
  const resizeStartXRef = useRef(0);
  const resizeStartWidthRef = useRef(CHAT_MIN_WIDTH);
  const forceChatScrollRef = useRef(false);
  const chatScrollRafRef = useRef<number | null>(null);
  const [currentPrice, setCurrentPrice] = useState(2154.30);
  const [priceChange, setPriceChange] = useState(-0.24);
  const [chatMessages, setChatMessages] = useState([
    { role: 'assistant', text: 'Chào bạn! Tôi là AI hỗ trợ giao dịch. Bạn cần giúp gì hôm nay?' }
  ]);



  // Model state
  const [models, setModels] = useState<{ id: string, name: string }[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [isModelLoading, setIsModelLoading] = useState(false);

  // Knowledge state
  const [knowledgeFiles, setKnowledgeFiles] = useState<any[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const scrollChatToBottom = (force = false) => {
    const el = chatMessagesRef.current;
    if (!el) return;

    if (chatScrollRafRef.current) cancelAnimationFrame(chatScrollRafRef.current);

    chatScrollRafRef.current = requestAnimationFrame(() => {
      const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
      if (!force && distanceFromBottom > 80) return;
      el.scrollTop = el.scrollHeight;
    });
  };



  useEffect(() => {
    if (sidebarView === 'ai') {
      scrollChatToBottom(forceChatScrollRef.current);
      forceChatScrollRef.current = false;
    }
  }, [chatMessages, sidebarView]);

  useEffect(() => {
    return () => {
      if (chatScrollRafRef.current) cancelAnimationFrame(chatScrollRafRef.current);
    };
  }, []);

  useEffect(() => {
    // Grid column changes don't always trigger a window resize event; notify the chart.
    setTimeout(() => window.dispatchEvent(new Event('resize')), 0);
  }, [sidebarView]);
  const [inputValue, setInputValue] = useState('');
  const [sessionId] = useState(() => Math.random().toString(36).substring(7));
  const modelsFetchedRef = useRef(false);

  useEffect(() => {
    const clamp = () => {
      const maxWidth = Math.floor(window.innerWidth * 0.5);
      setChatWidth(w => Math.max(CHAT_MIN_WIDTH, Math.min(w, maxWidth)));
    };
    clamp();
    window.addEventListener('resize', clamp);
    return () => window.removeEventListener('resize', clamp);
  }, []);

  useEffect(() => {
    if (!isChatResizing) return;

    const onMove = (e: MouseEvent) => {
      const dx = e.clientX - resizeStartXRef.current;
      const maxWidth = Math.floor(window.innerWidth * 0.5);
      const next = resizeStartWidthRef.current + dx;
      setChatWidth(Math.max(CHAT_MIN_WIDTH, Math.min(next, maxWidth)));
    };
    const onUp = () => setIsChatResizing(false);

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [isChatResizing]);

  // Fetch models on mount
  useEffect(() => {
    if (modelsFetchedRef.current) return;
    modelsFetchedRef.current = true;
    const fetchModels = async () => {
      try {
        const [modelsRes, currentModelRes] = await Promise.all([
          fetch(`${apiUrl}/api/chat/models`),
          fetch(`${apiUrl}/api/chat/current-model`)
        ]);

        if (modelsRes.ok && currentModelRes.ok) {
          const modelsData = await modelsRes.json();
          const currentData = await currentModelRes.json();
          setModels(modelsData);
          setSelectedModel(currentData.model_id);
        }
      } catch (err) {
        console.error("Failed to fetch models:", err);
      }
    };
    fetchModels();
  }, []);

  const handleModelChange = async (modelId: string) => {
    if (modelId === selectedModel || isModelLoading) return;

    setIsModelLoading(true);
    setSelectedModel(modelId);

    try {
      const response = await fetch(`${apiUrl}/api/chat/model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId })
      });

      if (!response.ok) throw new Error('Failed to switch model');

      setChatMessages(prev => [...prev, {
        role: 'assistant',
        text: `Đã chuyển sang mô hình: ${models.find(m => m.id === modelId)?.name || modelId}`
      }]);
    } catch (err) {
      console.error("Model switch error:", err);
      // Revert on failure
      fetch(`${apiUrl}/api/chat/current-model`)
        .then(res => res.json())
        .then(data => setSelectedModel(data.model_id));
    } finally {
      setIsModelLoading(false);
    }
  };

  const fetchKnowledgeFiles = async () => {
    try {
      const res = await fetch(`${apiUrl}/api/chat/knowledge/files`);
      if (res.ok) {
        const data = await res.json();
        setKnowledgeFiles(data);
      }
    } catch (err) {
      console.error("Failed to fetch knowledge files:", err);
    }
  };

  useEffect(() => {
    if (sidebarView === 'knowledge') {
      fetchKnowledgeFiles();
      const interval = setInterval(fetchKnowledgeFiles, 10000); // Poll every 10s for status
      return () => clearInterval(interval);
    }
  }, [sidebarView]);

  const uploadFile = async (file: File) => {
    if (isUploading) return;
    if (!file.name.endsWith('.pdf')) {
      alert('Chỉ hỗ trợ file PDF');
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${apiUrl}/api/chat/upload-knowledge`, {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        fetchKnowledgeFiles();
      } else {
        const err = await res.json();
        alert(`Lỗi: ${err.detail}`);
      }
    } catch (err) {
      console.error("Upload error:", err);
    } finally {
      setIsUploading(false);
    }
  };

  const handleUploadKnowledge = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) uploadFile(file);
    e.target.value = '';
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) uploadFile(file);
  };

  const handleDeleteKnowledgeFile = async (id: number) => {
    if (!window.confirm("Bạn có chắc chắn muốn xóa tài liệu này khỏi bộ não AI?")) return;

    try {
      const res = await fetch(`${apiUrl}/api/chat/knowledge/files/${id}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        fetchKnowledgeFiles();
      }
    } catch (err) {
      console.error("Delete error:", err);
    }
  };



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

      const startSimulation = (lastPoint: any, basePrice: number) => {
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

          // Calculate and set price change percentage
          const pChange = ((current.close - basePrice) / basePrice) * 100;
          setPriceChange(parseFloat(pChange.toFixed(2)));

          candlestickSeries.update(current);
        }, 1000);
      };

      const fetchData = async () => {
        try {
          const response = await fetch(`${apiUrl}/api/market-data`);
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const data = await response.json();
          if (Array.isArray(data) && data.length > 0) {
            candlestickSeries.setData(data);
            chart.timeScale().fitContent();

            const last = data[data.length - 1];
            const basePrice = data[0].open;
            setCurrentPrice(last.close);

            // Start live simulation based on last data point
            startSimulation(last, basePrice);
          } else {
            const mockData = generateMockData();
            candlestickSeries.setData(mockData);
            startSimulation(mockData[mockData.length - 1], mockData[0].open);
          }
        } catch (err) {
          const mockData = generateMockData();
          candlestickSeries.setData(mockData);
          startSimulation(mockData[mockData.length - 1], mockData[0].open);
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

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    forceChatScrollRef.current = true;
    const userMessage = { role: 'user', text: inputValue };
    const newMessages = [...chatMessages, userMessage];
    setChatMessages(newMessages);
    setInputValue('');

    // Add a placeholder for the assistant's response
    const assistantPlaceholder = { role: 'assistant', text: '' };
    setChatMessages(prev => [...prev, assistantPlaceholder]);

    try {
      const response = await fetch(`${apiUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: inputValue,
          session_id: sessionId,
          history: [] // History is now managed by the session on backend
        })
      });

      if (!response.ok) throw new Error('Failed to connect to AI');

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let fullAssistantText = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          fullAssistantText += chunk;

          // Update the last message in the set (the assistant response)
          setChatMessages(prev => {
            const updated = [...prev];
            if (updated.length > 0) {
              updated[updated.length - 1] = {
                role: 'assistant',
                text: fullAssistantText
              };
            }
            return updated;
          });
        }
      }


    } catch (err) {
      console.error('Chat error:', err);
      setChatMessages(prev => [
        ...prev.slice(0, -1),
        { role: 'assistant', text: 'Xin lỗi, tôi gặp lỗi kết nối với máy chủ AI. Vui lòng thử lại sau!' }
      ]);
    }
  };

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: (sidebarView === 'ai' || sidebarView === 'knowledge')
          ? `${NAV_WIDTH}px 0px minmax(0, 1fr) minmax(${ORDER_MIN_WIDTH}px, ${ORDER_MAX_WIDTH}px)`
          : `${NAV_WIDTH}px minmax(240px, ${MARKET_WIDTH}px) minmax(0, 1fr) minmax(${ORDER_MIN_WIDTH}px, ${ORDER_MAX_WIDTH}px)`,
        height: '100vh',
        width: '100%',
        position: 'relative',
        userSelect: isChatResizing ? 'none' : 'auto',
        cursor: isChatResizing ? 'ew-resize' : 'auto'
      }}
    >
      {/* Sidebar */}
      <nav style={styles.sidebar}>
        <div style={styles.logo}>TT</div>
        <NavItem active={sidebarView === 'market'} title="Thị trường" icon={<LayoutDashboard size={20} />} onClick={() => setSidebarView('market')} />
        <NavItem title="Biểu đồ" icon={<LineChart size={20} />} />
        <NavItem title="Danh mục" icon={<Briefcase size={20} />} />
        <NavItem title="Lịch sử" icon={<History size={20} />} />
        <NavItem active={sidebarView === 'ai'} title="AI support" icon={<MessageSquare size={20} />} onClick={() => setSidebarView('ai')} />
        <NavItem active={sidebarView === 'knowledge'} title="Quản lý kiến thức" icon={<Database size={20} />} onClick={() => setSidebarView('knowledge')} />

        <div style={{ marginTop: 'auto' }}>
          <NavItem title="Cài đặt" icon={<Settings size={20} />} />
        </div>
      </nav>

      {/* Second Column: Market or AI Chat */}
      <section style={styles.marketSidebar}>
        {sidebarView === 'market' && (
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
            </div>
          </>
        )}
      </section>

      {sidebarView === 'knowledge' && (
        <div style={{ ...styles.chatOverlay, width: chatWidth, left: NAV_WIDTH }}>
          <div
            style={styles.chatResizeHandle}
            onMouseDown={(e) => {
              setIsChatResizing(true);
              resizeStartXRef.current = e.clientX;
              resizeStartWidthRef.current = chatWidth;
            }}
          />
          <div style={styles.knowledgeContainer}>
            <div style={styles.marketHeader}>Quản lý Kiến thức (RAG)</div>
            
            <div
              style={{
                ...styles.uploadArea,
                ...(isDragging ? { borderColor: '#10B981', background: 'rgba(16,185,129,0.08)' } : {})
              }}
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragEnter={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
            >
              <label style={styles.uploadLabel}>
                <input
                  type="file"
                  accept=".pdf"
                  onChange={handleUploadKnowledge}
                  style={{ display: 'none' }}
                  disabled={isUploading}
                />
                {isUploading ? (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <Loader2 size={24} style={{ animation: 'spin 1s linear infinite' }} />
                    <span>Đang nạp dữ liệu...</span>
                  </div>
                ) : isDragging ? (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
                    <Upload size={32} color="#10B981" />
                    <span style={{ fontSize: '0.9em', color: '#10B981', fontWeight: 600 }}>Thả file vào đây!</span>
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
                    <Upload size={32} color="#10B981" />
                    <span style={{ fontSize: '0.9em', color: '#8B949E' }}>Click hoặc kéo thả PDF vào đây</span>
                  </div>
                )}
              </label>
            </div>

            <div style={styles.fileList}>
              <div style={{ fontSize: '0.8em', color: '#8B949E', marginBottom: '12px', fontWeight: 600 }}>TÀI LIỆU TRONG HỆ THỐNG</div>
              {knowledgeFiles.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '40px 0', color: '#484F58' }}>
                  <FileText size={48} style={{ marginBottom: '10px', opacity: 0.5 }} />
                  <div>Chưa có tài liệu nào</div>
                </div>
              ) : (
                knowledgeFiles.map(file => (
                  <div key={file.id} style={styles.fileItem}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1, minWidth: 0, overflow: 'hidden' }}>
                      <FileText size={20} color="#8B949E" style={{ flexShrink: 0 }} />
                      <div style={{ minWidth: 0, overflow: 'hidden' }}>
                        <div style={{ fontSize: '0.9em', fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {file.filename}
                        </div>
                        <div style={{ fontSize: '0.7em', color: '#8B949E' }}>
                          {new Date(file.created_at * 1000).toLocaleString()}
                        </div>
                      </div>
                    </div>
                    
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0, marginLeft: '8px' }}>
                      {file.status === 'completed' && (
                        <span style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.75em', color: '#10B981', background: 'rgba(16,185,129,0.1)', padding: '3px 8px', borderRadius: '12px', fontWeight: 600 }}>
                          <CheckCircle2 size={12} /> Đã xong
                        </span>
                      )}
                      {file.status === 'processing' && (
                        <span style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.75em', color: '#F59E0B', background: 'rgba(245,158,11,0.1)', padding: '3px 8px', borderRadius: '12px', fontWeight: 600 }}>
                          <Loader2 size={12} style={{ animation: 'spin 1s linear infinite' }} /> Đang nạp
                        </span>
                      )}
                      {file.status === 'error' && (
                        <span title={file.error_message} style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.75em', color: '#EF4444', background: 'rgba(239,68,68,0.1)', padding: '3px 8px', borderRadius: '12px', fontWeight: 600 }}>
                          <AlertCircle size={12} /> Lỗi
                        </span>
                      )}
                      
                      <button 
                        onClick={() => handleDeleteKnowledgeFile(file.id)}
                        style={styles.deleteBtn}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      {sidebarView === 'ai' && (
        <div style={{ ...styles.chatOverlay, width: chatWidth, left: NAV_WIDTH }}>
          <div
            style={styles.chatResizeHandle}
            onMouseDown={(e) => {
              setIsChatResizing(true);
              resizeStartXRef.current = e.clientX;
              resizeStartWidthRef.current = chatWidth;
            }}
          />
          <div style={styles.chatContainer}>
            <div style={{ ...styles.marketHeader, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>AI Support</span>
              <select
                value={selectedModel}
                onChange={(e) => handleModelChange(e.target.value)}
                disabled={isModelLoading || models.length === 0}
                style={{
                  background: '#0D1117',
                  color: '#10B981',
                  border: '1px solid #30363D',
                  borderRadius: '4px',
                  fontSize: '0.7em',
                  padding: '2px 4px',
                  outline: 'none',
                  cursor: isModelLoading || models.length === 0 ? 'not-allowed' : 'pointer',
                  maxWidth: '120px'
                }}
              >
                {models.length === 0 ? (
                  <option value="" disabled>Loading...</option>
                ) : (
                  models.map(m => (
                    <option key={m.id} value={m.id}>{m.name}</option>
                  ))
                )}
              </select>
            </div>



            <div ref={chatMessagesRef} style={styles.chatMessages}>
              {chatMessages.map((msg, i) => (
                <div key={i} style={{
                  ...styles.message,
                  alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                  background: msg.role === 'user' ? 'rgba(16, 185, 129, 0.2)' : '#0D1117'
                }}>
                  {msg.role === 'assistant' ? (
                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                      {normalizeMarkdownTables(msg.text)}
                    </ReactMarkdown>
                  ) : (
                    <span style={{ whiteSpace: 'pre-wrap' }}>{msg.text}</span>
                  )}
                </div>
              ))}
            </div>
            <div style={styles.chatInput}>
              <input
                type="text"
                placeholder={isModelLoading ? "Đang tải mô hình..." : "Hỏi AI..."}
                style={{ ...styles.ghostInput, opacity: isModelLoading ? 0.5 : 1 }}
                value={inputValue}
                disabled={isModelLoading}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
              />
              <Send
                size={16}
                style={{
                  cursor: isModelLoading ? 'not-allowed' : 'pointer',
                  color: isModelLoading ? '#30363D' : '#10B981'
                }}
                onClick={!isModelLoading ? handleSendMessage : undefined}
              />
            </div>
          </div>
        </div>
      )}

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

        <div ref={chartContainerRef} style={{ flex: 1, minWidth: 0, position: 'relative', background: '#0A0E14', overflow: 'hidden' }}>
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
  marketSidebar: {
    background: '#161B22',
    borderRight: '1px solid #30363D',
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    overflow: 'hidden',
    minWidth: 0
  },
  chatOverlay: {
    position: 'absolute',
    left: 60,
    top: 0,
    bottom: 0,
    background: '#161B22',
    borderRight: '1px solid #30363D',
    boxShadow: '0 10px 30px rgba(0,0,0,0.35)',
    zIndex: 20,
    display: 'flex',
    flexDirection: 'column'
  },
  chatResizeHandle: {
    position: 'absolute',
    right: -10,
    top: 0,
    bottom: 0,
    width: 12,
    cursor: 'ew-resize',
    background: 'transparent'
  },
  marketHeader: { padding: '20px', fontSize: '1em', fontWeight: 600 },
  marketTabs: { display: 'flex', padding: '0 20px 10px', gap: '15px', borderBottom: '1px solid #30363D' },
  marketTab: { fontSize: '0.8em', color: '#8B949E', cursor: 'pointer', paddingBottom: '5px' },
  marketTabActive: { color: '#E6EDF3', borderBottom: '2px solid #10B981' },
  searchBar: { padding: '15px 20px' },
  inputWrapper: { background: '#0D1117', border: '1px solid #30363D', padding: '8px 12px', borderRadius: '8px', display: 'flex', alignItems: 'center' },
  ghostInput: { background: 'transparent', border: 'none', color: 'white', outline: 'none', width: '100%', fontSize: '0.9em' },
  marketItems: { flex: 1, overflowY: 'auto' },
  marketItem: { display: 'flex', justifyContent: 'space-between', padding: '12px 20px', cursor: 'pointer', borderBottom: '1px solid rgba(48, 54, 61, 0.3)' },
  mainContent: { display: 'flex', flexDirection: 'column', minWidth: 0, overflow: 'hidden' },
  topNav: { height: '60px', background: '#161B22', borderBottom: '1px solid #30363D', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 20px' },
  pairInfo: { display: 'flex', alignItems: 'center', gap: '12px' },
  btnAi: { background: 'linear-gradient(135deg, #10B981 0%, #059669 100%)', border: 'none', padding: '5px 12px', borderRadius: '20px', color: 'white', fontSize: '0.75em', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', boxShadow: '0 4px 15px rgba(16, 185, 129, 0.2)' },
  accountInfo: { display: 'flex', alignItems: 'center', gap: '15px' },
  balanceBox: { background: 'rgba(255, 255, 255, 0.05)', padding: '5px 12px', borderRadius: '8px' },
  iconBtn: { color: '#8B949E', cursor: 'pointer' },
  orderPanel: { background: '#161B22', borderLeft: '1px solid #30363D', padding: '20px', display: 'flex', flexDirection: 'column', gap: '15px', minWidth: 0, overflow: 'hidden' },
  buySellToggle: { display: 'grid', gridTemplateColumns: '1fr 1fr', background: '#0D1117', padding: '4px', borderRadius: '10px' },
  toggleBtn: { background: 'transparent', border: 'none', color: '#8B949E', padding: '10px', borderRadius: '8px', fontWeight: 600, cursor: 'pointer' },
  toggleBtnBuy: { background: '#10B981', color: 'black' },
  toggleBtnSell: { background: '#EF4444', color: 'white' },
  inputGroup: { display: 'flex', flexDirection: 'column', gap: '6px' },
  label: { fontSize: '0.7em', color: '#8B949E', textTransform: 'uppercase' },
  footerBranding: {
    fontSize: '0.65em',
    color: '#30363D',
    marginTop: '20px',
    letterSpacing: '1px',
    textTransform: 'uppercase'
  },
  knowledgeContainer: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    background: '#0D1117',
    borderRight: '1px solid #30363D',
    height: '100%',
    padding: '20px',
  },
  uploadArea: {
    border: '2px dashed #30363D',
    borderRadius: '12px',
    padding: '40px 20px',
    textAlign: 'center',
    marginBottom: '24px',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    background: 'rgba(255, 255, 255, 0.02)',
  },
  uploadLabel: {
    cursor: 'pointer',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    width: '100%',
  },
  fileList: {
    flex: 1,
    overflowY: 'auto',
  },
  fileItem: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '12px',
    background: 'rgba(255, 255, 255, 0.03)',
    borderRadius: '8px',
    marginBottom: '8px',
    border: '1px solid transparent',
    transition: 'all 0.2s ease',
  },
  deleteBtn: {
    background: 'transparent',
    border: 'none',
    color: '#8B949E',
    cursor: 'pointer',
    padding: '6px',
    borderRadius: '4px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 0.2s ease',
    '&:hover': {
      color: '#EF4444',
      background: 'rgba(239, 68, 68, 0.1)',
    }
  },
  inputBox: { display: 'flex', alignItems: 'center', background: '#0D1117', border: '1px solid #30363D', padding: '10px 12px', borderRadius: '8px' },
  actionBtn: { width: '100%', padding: '14px', borderRadius: '12px', border: 'none', fontSize: '0.95em', fontWeight: 700, cursor: 'pointer', transition: '0.2s' },
  chatContainer: { display: 'flex', flexDirection: 'column', height: '100%' },
  chatMessages: { flex: 1, padding: '20px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px' },
  message: {
    padding: '10px 14px',
    borderRadius: '12px',
    fontSize: '0.85em',
    maxWidth: '85%',
    wordBreak: 'break-word',
    lineHeight: '1.5'
  },
  chatInput: { padding: '20px', borderTop: '1px solid #30363D', display: 'flex', gap: '10px', alignItems: 'center' }
};

export default App;
