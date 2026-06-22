import { useEffect, useState } from 'react';
import './index.css';
import DashboardMetrics from './components/DashboardMetrics';
import DashboardMap from './components/DashboardMap';
import DetectionsTable from './components/DetectionsTable';
import ChartView from './components/ChartView';

function App() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sourceFilter, setSourceFilter] = useState(''); // empty means all

  useEffect(() => {
    const fetchData = async () => {
      try {
        const url = sourceFilter ? `http://localhost:8000/detections?source=${sourceFilter}` : 'http://localhost:8000/detections';
        const response = await fetch(url);
        const json = await response.json();
        setData(json);
      } catch (error) {
        console.error('Error fetching detections:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [sourceFilter]);

  if (loading && data.length === 0) {
    return (
      <div className="app-container" style={{ alignItems: 'center', justifyContent: 'center', height: '100vh' }}>
        <h2 style={{ color: 'var(--text-secondary)' }}>Loading Intelligence Data...</h2>
      </div>
    );
  }

  return (
    <div className="app-container">
      <header className="header animated">
        <div>
          <h1>PedalRevolution Hub</h1>
          <p style={{ color: 'var(--text-secondary)', marginTop: '8px' }}>Tracking geospatial detections in real-time</p>
        </div>
        <div>
          <select 
            value={sourceFilter}
            onChange={(e) => setSourceFilter(e.target.value)}
            style={{ padding: '8px 16px', borderRadius: '8px', background: 'var(--bg-card)', color: '#fff', border: '1px solid var(--glass-border)', outline: 'none' }}
          >
            <option value="">All Sources</option>
            <option value="online">Online App Detections</option>
            <option value="gt">Ground Truth Detections</option>
          </select>
        </div>
      </header>
      
      <DashboardMetrics data={data} className="animated delay-1" />
      
      <div className="dashboard-row">
        <div className="glass-panel animated delay-2" style={{ padding: '0' }}>
          <DashboardMap data={data} />
        </div>
        <div className="glass-panel animated delay-2">
          <ChartView data={data} />
        </div>
      </div>
      
      <div className="glass-panel animated delay-3">
        <DetectionsTable data={data} />
      </div>
    </div>
  );
}

export default App;
