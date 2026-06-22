import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix for default marker icons in React Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

export default function DashboardMap({ data }) {
  const geoData = data.filter(d => d.latitude !== null && d.longitude !== null);
  
  // Default to a central European or US coordinate if no hits, else center to the last detection
  const centerPosition = geoData.length > 0 
    ? [geoData[geoData.length - 1].latitude, geoData[geoData.length - 1].longitude]
    : [37.7749, -122.4194];

  return (
    <div className="map-wrapper" style={{ height: '100%' }}>
      <MapContainer 
        center={centerPosition} 
        zoom={13} 
        style={{ height: '100%', width: '100%', background: '#0f172a' }}
      >
        <TileLayer
          attribution='&copy; OpenStreetMap'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />
        {geoData.map(detection => (
          <Marker 
            key={detection.id} 
            position={[detection.latitude, detection.longitude]}
          >
            <Popup>
              <strong>{detection.label}</strong> ({(detection.confidence * 100).toFixed(1)}%)<br/>
              Time: {new Date(detection.timestamp_ms).toLocaleTimeString()}
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
}
