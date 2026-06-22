import { Activity, Crosshair, MapPin } from 'lucide-react';

export default function DashboardMetrics({ data, className = '' }) {
  const total = data.length;
  const avgConf = data.length > 0 
    ? (data.reduce((acc, curr) => acc + curr.confidence, 0) / total * 100).toFixed(1) 
    : 0;
  const geoTagged = data.filter(d => d.latitude !== null).length;

  return (
    <div className={`metrics-grid ${className}`}>
      <div className="glass-panel metric-card">
        <div className="metric-title">
          <Activity size={18} color="var(--accent-primary)" />
          Total Detections
        </div>
        <div className="metric-value">{total}</div>
      </div>
      
      <div className="glass-panel metric-card">
        <div className="metric-title">
          <Crosshair size={18} color="var(--accent-secondary)" />
          Average Confidence
        </div>
        <div className="metric-value">{avgConf}%</div>
      </div>

      <div className="glass-panel metric-card">
        <div className="metric-title">
          <MapPin size={18} color="var(--success)" />
          Geo-Tagged Objects
        </div>
        <div className="metric-value">{geoTagged}</div>
      </div>
    </div>
  );
}
