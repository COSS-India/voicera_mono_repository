# Frontend Service

Comprehensive documentation for the VoicEra Frontend service.

## Overview

The Frontend is the user-facing web interface for VoicEra, built with **Next.js 16+**, **React 18+**, and **TailwindCSS 4+**.

**Key Responsibilities:**
- User authentication & account management
- Agent creation and management
- Campaign management and monitoring
- Real-time voice call interface
- Call history and recordings
- Analytics dashboard

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

```bash
cd voicera_frontend

# Install dependencies
npm install
# or
yarn install

# Configure environment
cp .env.example .env.local
# Edit with your settings
```

### Development

```bash
# Run dev server (with hot reload)
npm run dev

# Open in browser
# http://localhost:3000
```

### Production Build

```bash
# Build for production
npm run build

# Start production server
npm run start

# Or via Docker
docker build -t voicera-frontend .
docker run -p 3000:3000 voicera-frontend
```

---

## Project Structure

```
voicera_frontend/
├── app/
│   ├── layout.tsx             # Root layout
│   ├── page.tsx               # Home page
│   ├── (auth)/                # Auth routes
│   │   ├── login/
│   │   │   └── page.tsx
│   │   ├── signup/
│   │   │   └── page.tsx
│   │   └── forgot-password/
│   │       └── page.tsx
│   ├── (dashboard)/           # Protected routes
│   │   ├── layout.tsx
│   │   ├── page.tsx           # Dashboard home
│   │   ├── agents/
│   │   │   ├── page.tsx       # Agents list
│   │   │   ├── [id]/
│   │   │   │   └── page.tsx   # Agent details
│   │   │   └── new/
│   │   │       └── page.tsx   # Create agent
│   │   ├── campaigns/
│   │   │   ├── page.tsx
│   │   │   ├── [id]/
│   │   │   │   └── page.tsx
│   │   │   └── new/
│   │   │       └── page.tsx
│   │   ├── call-logs/
│   │   │   ├── page.tsx
│   │   │   └── [id]/
│   │   │       └── page.tsx
│   │   ├── analytics/
│   │   │   └── page.tsx
│   │   ├── settings/
│   │   │   └── page.tsx
│   │   └── profile/
│   │       └── page.tsx
│   └── api/                   # API routes (if needed)
│       ├── auth/
│       └── proxy/
├── components/
│   ├── app-sidebar.tsx        # Navigation sidebar
│   ├── assistants/            # Agent components
│   │   ├── agent-form.tsx
│   │   ├── agent-card.tsx
│   │   └── agent-list.tsx
│   ├── campaigns/             # Campaign components
│   │   ├── campaign-form.tsx
│   │   ├── campaign-card.tsx
│   │   └── campaign-list.tsx
│   ├── voice/                 # Voice call components
│   │   ├── voice-interface.tsx
│   │   ├── audio-player.tsx
│   │   └── call-status.tsx
│   ├── analytics/             # Analytics components
│   │   ├── metrics-card.tsx
│   │   ├── chart-widget.tsx
│   │   └── analytics-dashboard.tsx
│   ├── ui/                    # Reusable UI components
│   │   ├── button.tsx
│   │   ├── card.tsx
│   │   ├── modal.tsx
│   │   ├── dropdown.tsx
│   │   ├── input.tsx
│   │   └── ...
│   ├── header.tsx
│   ├── footer.tsx
│   └── layouts/
│       ├── authenticated-layout.tsx
│       └── public-layout.tsx
├── hooks/
│   ├── use-auth.ts            # Auth hook
│   ├── use-voice.ts           # Voice call hook
│   ├── use-api.ts             # API interaction
│   ├── use-mobile.ts          # Mobile detection
│   └── use-analytics.ts
├── lib/
│   ├── api.ts                 # API client
│   ├── api-config.ts          # API configuration
│   ├── auth.ts                # Auth utilities
│   ├── websocket.ts           # WebSocket client
│   ├── utils.ts               # Utility functions
│   └── constants.ts           # Constants
├── styles/
│   └── globals.css
├── public/                    # Static assets
│   └── images/
├── package.json
├── tsconfig.json
├── next.config.ts
├── tailwind.config.js
├── postcss.config.js
└── .env.example
```

---

## Key Features

### Authentication

```typescript
// Custom hook for auth state
function useAuth() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    // Check if user is logged in
    const token = localStorage.getItem('token');
    if (token) {
      verifyToken(token).then(setUser);
    }
    setLoading(false);
  }, []);
  
  const login = async (email, password) => {
    const response = await api.post('/auth/login', {
      email,
      password
    });
    localStorage.setItem('token', response.token);
    setUser(response.user);
  };
  
  const logout = () => {
    localStorage.removeItem('token');
    setUser(null);
  };
  
  return { user, loading, login, logout };
}
```

### Voice Call Interface

```typescript
// Component for real-time voice calls
function VoiceInterface({ campaignId, agentId }) {
  const [callActive, setCallActive] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [isListening, setIsListening] = useState(false);
  const mediaRecorder = useRef(null);
  const ws = useRef(null);
  
  const startCall = async () => {
    // Establish WebSocket connection
    ws.current = new WebSocket(
      process.env.NEXT_PUBLIC_VOICE_SERVER_URL
    );
    
    ws.current.onopen = async () => {
      // Send auth token
      ws.current.send(JSON.stringify({
        type: 'auth',
        token: getToken(),
        agent_id: agentId
      }));
    };
    
    ws.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      if (message.type === 'ready') {
        setCallActive(true);
        startAudioCapture();
      } else if (message.type === 'audio') {
        playAudio(message.data);
      }
    };
  };
  
  const startAudioCapture = async () => {
    const stream = await navigator.mediaDevices
      .getUserMedia({ audio: true });
    mediaRecorder.current = new MediaRecorder(stream);
    
    mediaRecorder.current.ondataavailable = (event) => {
      const audioData = event.data;
      ws.current.send(JSON.stringify({
        type: 'audio',
        data: audioData
      }));
    };
    
    mediaRecorder.current.start(100); // Send chunks every 100ms
    setIsListening(true);
  };
  
  const endCall = () => {
    mediaRecorder.current.stop();
    ws.current.send(JSON.stringify({
      type: 'control',
      action: 'end'
    }));
    ws.current.close();
    setCallActive(false);
    setIsListening(false);
  };
  
  return (
    <div className="voice-interface">
      <div className="call-status">
        {callActive ? 'Call in Progress' : 'Call Ended'}
      </div>
      
      <div className="transcript">
        <p>{transcript}</p>
      </div>
      
      <div className="controls">
        {!callActive ? (
          <button onClick={startCall} className="btn-primary">
            Start Call
          </button>
        ) : (
          <button onClick={endCall} className="btn-danger">
            End Call
          </button>
        )}
      </div>
      
      <div className="audio-status">
        {isListening && (
          <span className="recording-indicator">
            🎤 Recording...
          </span>
        )}
      </div>
    </div>
  );
}
```

### Analytics Dashboard

```typescript
// Analytics component
function AnalyticsDashboard() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const fetchMetrics = async () => {
      const response = await api.get('/analytics/calls');
      setMetrics(response.data);
      setLoading(false);
    };
    
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);
  
  if (loading) return <LoadingSpinner />;
  
  return (
    <div className="analytics-grid">
      <MetricCard
        title="Total Calls"
        value={metrics.total_calls}
        icon="📞"
      />
      
      <MetricCard
        title="Average Duration"
        value={`${Math.round(metrics.avg_duration)}s`}
        icon="⏱️"
      />
      
      <MetricCard
        title="Sentiment Score"
        value={`${metrics.sentiment_positive}%`}
        icon="😊"
      />
      
      <ChartWidget
        title="Calls Over Time"
        data={metrics.calls_by_hour}
        type="line"
      />
      
      <ChartWidget
        title="Sentiment Distribution"
        data={metrics.sentiment_distribution}
        type="pie"
      />
    </div>
  );
}
```

---

## API Integration

### API Client Configuration

```typescript
// lib/api-config.ts
export const API_BASE_URL = 
  process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

export const API_TIMEOUT = 30000;

export const API_ENDPOINTS = {
  AUTH: {
    LOGIN: '/auth/login',
    SIGNUP: '/auth/signup',
    REFRESH: '/auth/refresh-token',
    ME: '/auth/me'
  },
  AGENTS: {
    LIST: '/agents',
    CREATE: '/agents',
    DETAIL: (id: string) => `/agents/${id}`,
    UPDATE: (id: string) => `/agents/${id}`,
    DELETE: (id: string) => `/agents/${id}`
  },
  CAMPAIGNS: {
    LIST: '/campaigns',
    CREATE: '/campaigns',
    DETAIL: (id: string) => `/campaigns/${id}`,
    LAUNCH: (id: string) => `/campaigns/${id}/launch`
  },
  CALL_LOGS: {
    LIST: '/call-logs',
    DETAIL: (id: string) => `/call-logs/${id}`
  },
  ANALYTICS: {
    CALLS: '/analytics/calls',
    SENTIMENT: '/analytics/sentiment'
  }
};
```

### API Client

```typescript
// lib/api.ts
import axios from 'axios';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT
});

// Add token to headers
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle errors
apiClient.interceptors.response.use(
  response => response,
  error => {
    if (error.response?.status === 401) {
      // Redirect to login
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const api = {
  get: (url: string) => apiClient.get(url),
  post: (url: string, data: any) => apiClient.post(url, data),
  put: (url: string, data: any) => apiClient.put(url, data),
  delete: (url: string) => apiClient.delete(url)
};
```

---

## Styling

### TailwindCSS Configuration

```typescript
// tailwind.config.js
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}'
  ],
  theme: {
    extend: {
      colors: {
        primary: '#3B82F6',
        secondary: '#10B981',
        danger: '#EF4444'
      },
      spacing: {
        'container': '1200px'
      }
    }
  },
  plugins: []
};
```

### Example Component Styling

```typescript
export function AgentCard({ agent }) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
      <h3 className="text-lg font-semibold text-gray-900">
        {agent.name}
      </h3>
      
      <p className="text-sm text-gray-600 mt-2">
        {agent.description}
      </p>
      
      <div className="flex gap-2 mt-4">
        <button className="px-4 py-2 bg-primary text-white rounded hover:bg-blue-600">
          Edit
        </button>
        <button className="px-4 py-2 bg-danger text-white rounded hover:bg-red-600">
          Delete
        </button>
      </div>
    </div>
  );
}
```

---

## Environment Variables

```env
# API Configuration
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_API_TIMEOUT=30000

# Voice Server
NEXT_PUBLIC_VOICE_SERVER_URL=http://localhost:7860
NEXT_PUBLIC_WS_URL=ws://localhost:7860

# Authentication
NEXT_PUBLIC_AUTH_ENABLED=true
NEXT_PUBLIC_JWT_STORAGE_KEY=voicera_token

# Application
NEXT_PUBLIC_APP_NAME=VoicEra
NEXT_PUBLIC_APP_VERSION=1.0.0
NEXT_PUBLIC_LOG_LEVEL=info

# Analytics
NEXT_PUBLIC_ANALYTICS_ENABLED=false
```

---

## Next Steps

- **[Installation](../getting-started/installation.md)** - Set up development
- **[Quick Start](../getting-started/quickstart.md)** - Get running
- **[Backend API](backend.md)** - API documentation
