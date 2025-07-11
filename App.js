import React, { useEffect, useState } from 'react';
import Dashboard from './components/Dashboard';
import Alert from './components/Alert';

function App() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/predict')
      .then(res => res.json())
      .then(json => setData(json));
  }, []);

  return (
    <div className="p-4">
      {data && data.alert && <Alert score={data.strain_score} />}
      <Dashboard data={data} />
    </div>
  );
}

export default App;
