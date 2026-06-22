export default function DetectionsTable({ data }) {
  return (
    <div className="table-container">
      <div className="metric-title" style={{ marginBottom: '16px', padding: '0 8px' }}>Recent Detections</div>
      <table>
        <thead>
          <tr>
            <th>Time</th>
            <th>Type</th>
            <th>Confidence</th>
            <th>Coordinates</th>
          </tr>
        </thead>
        <tbody>
          {data.slice(-10).reverse().map(d => (
            <tr key={d.id}>
              <td>{new Date(d.timestamp_ms).toLocaleTimeString()}</td>
              <td><span className="badge">{d.label}</span></td>
              <td>{(d.confidence * 100).toFixed(1)}%</td>
              <td style={{ color: 'var(--text-secondary)' }}>
                {d.latitude ? `${d.latitude.toFixed(4)}, ${d.longitude.toFixed(4)}` : 'N/A'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {data.length === 0 && (
        <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-secondary)' }}>
          No detections yet
        </div>
      )}
    </div>
  );
}
