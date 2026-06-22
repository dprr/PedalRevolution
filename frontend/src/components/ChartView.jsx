import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function ChartView({ data }) {
  // Aggregate vehicle labels
  const dist = data.reduce((acc, curr) => {
    acc[curr.label] = (acc[curr.label] || 0) + 1;
    return acc;
  }, {});

  const chartData = Object.keys(dist).map(key => ({
    name: key,
    count: dist[key]
  }));

  const colors = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ec4899'];

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="metric-title" style={{ marginBottom: '16px' }}>Label Distribution</div>
      <div style={{ flexGrow: 1, minHeight: '300px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData}>
            <XAxis dataKey="name" stroke="#94a3b8" tick={{ fill: '#94a3b8' }} />
            <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8' }} />
            <Tooltip 
              cursor={{ fill: 'rgba(255, 255, 255, 0.05)' }} 
              contentStyle={{ background: '#1e293b', border: 'none', borderRadius: '8px', color: '#f8fafc' }}
            />
            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
