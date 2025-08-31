import React, { useEffect, useRef, useState } from 'react';
import { useStatePersistence } from './utils/statePersistence';
import DOMPurify from 'dompurify';
import securityMonitor from './utils/securityMonitor'; // 🛡️ SECURITY MONITORING
import dynamicLoader from './utils/dynamicLoader'; // 🎭 DYNAMIC SECURITY LOADER
import { SecurityProvider } from './contexts/SecurityContext'; // 🛡️ SECURITY CONTEXT
import SecurityWrapper from './components/SecurityWrapper'; // 🛡️ SECURITY WRAPPER  
import AlkinehotelShell from './components/AlkinehotelShell'; // 🍯 Alkinehotel SHELL
import LoginPage from './components/LoginPage'; // 🔐 PROFESSIONAL LOGIN PAGE

//  EXTRACTED HOOKS - MUCH CLEANER!
import {
  useApi,
  useAppState,
  useAuth,
  useChat,
  useModels,
  useConfigManagement,
  useTabState,
  useDrumcircle,
  useThreading,
  useWebSocket
} from './hooks';
import { useSecurityContext } from './contexts/SecurityContext'; // 🛡️ For role-based access control

//  EXTRACTED TAB CONTENT RENDERER
import SecureTabRenderer from './components/SecureTabRenderer'; // 🛡️ SECURE TAB RENDERING
import ErrorBoundary from './components/ErrorBoundary';

// 🎨 EXTRACTED TAB STYLES
import './styles/TabStyles.css';
import './App.css';
import './compact.css';
import 'highlight.js/styles/github-dark.css';

function App() {
  // 🔐 AUTHENTICATION LOGIC NOW HANDLED BY SECURITYWRAPPER
  // Remove direct SecurityContext usage to prevent white screen

  // 🂂 ROLE-BASED TAB FILTERING COMPONENT
  const RoleBasedTabs = ({ activeTab, setActiveTab }) => {
    // Get user access level from SecurityContext (safely)
    let userAccessLevel = 'friend'; // Default to friend level
    let securityContext = null;
    let isAuthenticated = false;
    
    try {
      securityContext = useSecurityContext();
      userAccessLevel = securityContext?.accessLevel || 'friend';
      isAuthenticated = securityContext?.isAuthenticated || false;
      
      console.log('🔐 RoleBasedTabs - SecurityContext state:');
      console.log('  isAuthenticated:', isAuthenticated);
      console.log('  accessLevel:', userAccessLevel);
      console.log('  securityMode:', securityContext?.securityMode);
    } catch (error) {
      // SecurityContext not available in this render, use default
      console.log('SecurityContext not available, using friend level');
    }

    // Define all tabs with their minimum access levels
    const allTabs = [
      { id: 'chat', label: '💬 Chat', icon: '💬', minLevel: 'friend' },
      { id: 'drumcircle', label: '🥁 Drumcircle', icon: '🥁', minLevel: 'friend' },
      { id: 'threaded', label: '🧵 Threaded', icon: '🧵', minLevel: 'friend' },
      { id: 'web-search', label: '🌐 Web Search', icon: '🌐', minLevel: 'friend' },
      { id: 'phi-detection', label: '🔒 PHI Detection', icon: '🔒', minLevel: 'friend' },
      { id: 'python-magic', label: '🐍 Python Magic', icon: '🐍', minLevel: 'friend' },
      { id: 'sessions', label: '💼 Sessions', icon: '💼', minLevel: 'friend' },
      { id: 'llm', label: '🤖 LLM Providers', icon: '🤖', minLevel: 'friend' }, //  RESTORED for friends!
      { id: 'config', label: '⚙️ Configuration', icon: '⚙️', minLevel: 'friend' },
      { id: 'debug', label: '🐛 Debug', icon: '🐛', minLevel: 'admin' } // 🔒 ADMIN ONLY
    ];

    // Filter tabs based on access level
    const visibleTabs = allTabs.filter(tab => {
      if (userAccessLevel === 'admin') return true; // Admin sees everything
      return tab.minLevel === 'friend'; // Friend only sees friend-level tabs
    });

    // Add access level indicator for debugging
    console.log(`🔐 User access level: ${userAccessLevel}, visible tabs: ${visibleTabs.length}`);

    return (
      <>
        {visibleTabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
          >
            {tab.label}
          </button>
        ))}
        {/* 🔍 DEBUG: Show access level and backend status */}
        <div style={{ 
          fontSize: '12px',
          color: '#666',
          marginLeft: '10px',
          display: 'flex',
          gap: '10px',
          alignItems: 'center'
        }}>
          <span style={{ 
            padding: '4px 8px',
            backgroundColor: userAccessLevel === 'admin' ? '#e8f5e8' : '#fff3cd',
            borderRadius: '4px'
          }}>
            {userAccessLevel === 'admin' ? '🔓 Admin' : '👥 Friend'}
          </span>
          
          <span style={{ 
            padding: '4px 8px',
            backgroundColor: backendReady ? '#e8f5e8' : '#ffe8e8',
            borderRadius: '4px',
            border: `1px solid ${backendReady ? '#4CAF50' : '#f44336'}`
          }}>
            {backendReady ? '🟢 Connected' : '🔴 Connecting...'}
          </span>
        </div>
      </>
    );
  };

  // 🔍 ROLE-BASED ACCESS CONTROL
  const SecurityAwareComponent = ({ children }) => {
    const { accessLevel } = useSecurityContext();
    return React.cloneElement(children, { userAccessLevel: accessLevel });
  };

  // 🎭 SIMPLE ROUTING FOR Alkinehotel PAGES
  const [currentRoute] = useState(() => {
    return window.location.pathname;
  });

  // 🍯 RENDER Alkinehotel SHELL IF REQUESTED (NO AUTH NEEDED - IT'S A TRAP!)
  if (currentRoute === '/alkinehotel-shell') {
    return <AlkinehotelShell />;
  }

  // ️ AUTHENTICATION NOW HANDLED BY SECURITYWRAPPER

  // Helper function to load YAML config from string
  const loadContinueConfigFromYaml = async (yamlContent) => {
    if (!yamlContent.trim()) return;

    try {
      const API_URL = `http://${remoteHost}:${config.port}`;
      const response = await fetch(`${API_URL}/load-continue-config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ yaml_content: yamlContent }),
      });
      const result = await response.json();
      if (result.status === 'config_loaded') {
        console.log('✅ YAML config auto-loaded successfully');
        // Refresh available models
        const modelsResponse = await fetch(`${API_URL}/available-models`);
        const modelsData = await modelsResponse.json();
        setAvailableModels(modelsData);
      }
    } catch (error) {
      console.error('❌ Auto-load YAML config failed:', error);
    }
  };
  // 💾 UNIFIED STATE PERSISTENCE
  const { saveTabState, loadTabState, loadAllState } = useStatePersistence();

  //  EXTRACTED STATE MANAGEMENT - MUCH CLEANER!
  const { config, setConfig, apiKeys, setApiKeys, remoteHost, setRemoteHost, backendReady, loadSavedConfig, updateApiKeys, updateConfigDebounced, updateConfig, autoSaveAllConfig } = useConfigManagement();
  const { activeTab, setActiveTab } = useTabState();

  // 🔥 DRUMCIRCLE STATE from useAppState  
  const {
    drumcircleTaskType, setDrumcircleTaskType,
    drumcircleRounds, setDrumcircleRounds,
    includeGemini, setIncludeGemini, drumcircleChaosMode, drumcircleTemperature, setDrumcircleTemperature, drumcircleResponseLength, setDrumcircleResponseLength, drumcircleOutputFormat, setDrumcircleOutputFormat, drumcircleStreamingMode, setDrumcircleStreamingMode, drumcircleTimeout, setDrumcircleTimeout, drumcircleDebugMode, setDrumcircleDebugMode, drumcircleRetryFailures, setDrumcircleRetryFailures,
    selectedDrumcircleModels, setSelectedDrumcircleModels, showHistory, setShowHistory, consoleMaxEntries, setConsoleMaxEntries, editorRef, setDrumcircleChaosMode
  } = useAppState();

  // 🥁 NEW: drumcircleRoleMapping state
  const [drumcircleRoleMapping, setDrumcircleRoleMapping] = useState({});
  const [isRichText, setIsRichText] = useState(true);
  const [editorType, setEditorType] = useState('plain'); // 🔥 'plain', 'tinymce', 'quill', 'monaco'
  // 🤖 CHAT & MODELS STATE
  const [lastUserPrompt, setLastUserPrompt] = useState('');
  // messages, conversationHistory, contextEnabled, maxContextMessages now come from useChat hook
  const [availableModels, setAvailableModels] = useState({});
  const [selectedModel, setSelectedModel] = useState('');
  const [llmProviders, setLlmProviders] = useState([]);
  const [configYaml, setConfigYaml] = useState('');

  // 🌐 WEBSOCKET STATE
  const [connected, setConnected] = useState(false);
  const [peers, setPeers] = useState([]);
  const wsRef = useRef(null);

  // 🤖 CHAT FUNCTIONS - Get from a single useChat call (MOVED BEFORE WEBSOCKET)
  const {
    chatWithModel,
    toggleContext,
    clearConversation,
    fetchConversationHistory,
    conversationHistory,
    setConversationHistory,
    messages,
    setMessages,
    pendingMessages, // 🚀 NEW: Pending messages with status
    contextEnabled,
    setContextEnabled,
    maxContextMessages,
    setMaxContextMessages
  } = useChat(
    remoteHost,
    config.port,
    selectedModel,
    isRichText,
    config
  );

  // 🌐 WEBSOCKET COMMUNICATION (MOVED AFTER CHAT TO ACCESS setMessages)
  const { sendMessage, sendLiveUpdate } = useWebSocket({
    remoteHost,
    config,
    setConnected,
    setMessages,
    setPeers,
    wsRef
  });
  // showHistory and setShowHistory come from useAppState, not useChat

  // 🥁 DRUMCIRCLE FUNCTIONALITY
  const { startDrumcircle: startDrumcircleBase, startConfigurableDrumcircle: startConfigurableDrumcircleBase } = useDrumcircle(remoteHost, config.port);

  // Wrap the drumcircle functions to pass the current state
  const startDrumcircle = (customPrompt = null) => {
    return startDrumcircleBase(customPrompt, {
      includeGemini,
      drumcircleTaskType,
      drumcircleTemperature,
      drumcircleTimeout,
      drumcircleDebugMode
    });
  };

  const startConfigurableDrumcircle = (customPrompt = null) => {
    return startConfigurableDrumcircleBase(customPrompt, {
      drumcircleRoleMapping,
      drumcircleTaskType,
      drumcircleRounds,
      includeGemini,
      drumcircleTemperature,
      drumcircleResponseLength,
      drumcircleOutputFormat,
      drumcircleStreamingMode,
      drumcircleTimeout,
      drumcircleDebugMode,
      drumcircleRetryFailures,
      drumcircleChaosMode
    });
  };

  // 🧵 THREADING FUNCTIONALITY  
  const {
    startThreadedConversation: startThreadedConversationBase,
    fetchActiveThreads: fetchActiveThreadsBase,
    viewThread: viewThreadBase
  } = useThreading(remoteHost, config.port);

  // Threading state is managed directly here (not in the hook)
  const [threadPrompt, setThreadPrompt] = useState('');
  const [selectedThreadModels, setSelectedThreadModels] = useState([]);
  const [maxThreadRounds, setMaxThreadRounds] = useState(3);
  const [activeThreads, setActiveThreads] = useState([]);
  const [selectedThread, setSelectedThread] = useState(null);

  // Wrap threading functions to pass state and update local state
  const startThreadedConversation = async () => {
    try {
      if (!threadPrompt || !selectedThreadModels || selectedThreadModels.length < 2) {
        console.warn('⚠️ Cannot start threading: Need prompt and at least 2 models');
        return;
      }

      return await startThreadedConversationBase({
        threadPrompt,
        selectedThreadModels,
        maxThreadRounds,
        onSuccess: () => fetchActiveThreads() // Refresh threads on success
      });
    } catch (error) {
      console.error('❌ Failed to start threaded conversation:', error);
      alert('Failed to start threaded conversation: ' + error.message);
    }
  };

  const fetchActiveThreads = async () => {
    try {
      const threads = await fetchActiveThreadsBase();
      setActiveThreads(threads || []);
      return threads;
    } catch (error) {
      console.error('❌ Failed to fetch active threads:', error);
      setActiveThreads([]);
      return [];
    }
  };

  const viewThread = async (threadId) => {
    try {
      const thread = await viewThreadBase(threadId);
      if (thread) {
        setSelectedThread(thread);
      }
      return thread;
    } catch (error) {
      console.error('❌ Failed to view thread:', error);
      alert('Failed to load thread: ' + error.message);
      return null;
    }
  };

  const handlePaste = (e) => {
    if (!isRichText) return;

    e.preventDefault();
    const html = e.clipboardData.getData('text/html');
    const text = e.clipboardData.getData('text/plain');

    if (html) {
      document.execCommand('insertHTML', false, DOMPurify.sanitize(html));
    } else {
      document.execCommand('insertText', false, text);
    }
  };

  // 🦙 OLLAMA MODEL TESTING FUNCTION
  const handleTestOllamaModels = async () => {
    try {
      console.log('🔬 Starting Ollama model testing...');

      // 🔒 DEFENSIVE: Preserve config state during testing
      const safePort = config.port || 7878;
      const safeHost = remoteHost || 'localhost';

      console.log('🔧 Using safe config:', { host: safeHost, port: safePort });

      const API_URL = `http://${safeHost}:${safePort}`;
      const response = await fetch(`${API_URL}/test-ollama-models`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          verbosity: 2, // Detailed verbosity for now
          test_prompt: "Hello! Please respond with 'OK' to confirm you are working."
        }),
      });

      const result = await response.json();

      if (result.status === 'success') {
        console.log(`🎉 Testing complete! ${result.results.filter(r => r.status === 'passed').length}/${result.total_models} models passed`);

        // Show results in a nice format
        console.table(result.results.map(r => ({
          Model: r.model,
          Status: r.status,
          Latency: r.latency_ms ? `${r.latency_ms}ms` : 'N/A',
          Preview: r.response_preview || r.error || 'N/A'
        })));

        alert(`🦙 Testing Results:\n\n✅ Passed: ${result.results.filter(r => r.status === 'passed').length}\n❌ Failed: ${result.results.filter(r => r.status === 'failed').length}\n\nCheck console for detailed results!`);
      } else {
        console.error('❌ Testing failed:', result.message);
        alert(`❌ Model testing failed: ${result.message}`);
      }
    } catch (error) {
      console.error('❌ Failed to test models:', error);
      alert(`❌ Error testing models: ${error.message}`);
    }
  };

  // updateConfig and updateConfigDebounced are now properly extracted from useConfigManagement hook

  const handleInput = (e) => {
    const content = e.target.innerHTML || e.target.innerText || '';
    // setInputContent(content); // inputContent is not defined in App.jsx

    // Debounce logging to reduce spam
    // if (inputTimeoutRef.current) { // inputTimeoutRef is not defined in App.jsx
    //   clearTimeout(inputTimeoutRef.current);
    // }
    // inputTimeoutRef.current = setTimeout(() => {
    //   console.log(`📝 TEXT INPUT: [${content.length} chars]`);
    // }, 500);

    // if (config.live_mode && wsRef.current) {
    //   console.log('📡 LIVE MODE: Sending live update');
    //   const message = {
    //     type: 'LiveUpdate',
    //     id: crypto.randomUUID(),
    //     content,
    //     is_html: isRichText,
    //     cursor_position: null,
    //   };

    //   try {
    //     wsRef.current.send(JSON.stringify(message));
    //     console.log('✅ Live update sent');
    //   } catch (error) {
    //     console.error('❌ Failed to send live update:', error);
    //   }
    // }
  };

  // 💾 Load saved state on mount - REWRITTEN TO USE BACKEND AS SINGLE SOURCE OF TRUTH
  useEffect(() => {
    const loadBackendData = async () => {
      const backendData = await loadSavedConfig();
      if (backendData) {
        console.log('✅ Applying UI preferences from backend:', backendData.ui_preferences);
        
        // Apply UI preferences from the backend
        if (backendData.ui_preferences) {
          const prefs = backendData.ui_preferences;
          setSelectedModel(prefs.selected_model || '');
          setContextEnabled(prefs.context_enabled !== undefined ? prefs.context_enabled : true);
          setMaxContextMessages(prefs.max_context_messages || 20);
          setActiveTab(prefs.active_tab || 'chat');
          setSelectedDrumcircleModels(prefs.selected_drumcircle_models || []);
          setDrumcircleTaskType(prefs.drumcircle_task_type || 'debate');
          setDrumcircleRounds(prefs.drumcircle_rounds || 1);
          setIncludeGemini(prefs.include_gemini || false);
        }

        // Apply LLM config YAML from the backend
        if (backendData.llm_config_yaml) {
          console.log('✅ YAML RESTORED FROM BACKEND:', backendData.llm_config_yaml.length, 'chars');
          setConfigYaml(backendData.llm_config_yaml);
          // Load LLM config from YAML
          loadContinueConfigFromYaml(backendData.llm_config_yaml);
        }
      }
    };

    if (backendReady) {
        loadBackendData();
    }
  }, [backendReady]); // This effect runs once when the backend becomes ready

  // 💾 REMOVED: The useEffect hooks that called saveTabState to save to localStorage are gone.
  // The autoSaveAllConfig hook below is now the only save mechanism.

  // 💾 AUTO-SAVE: Backend config (separate from local state)
  useEffect(() => {
    const timed = setTimeout(() => {
      autoSaveAllConfig({
        config: config,
        api_keys: apiKeys,
        ui_preferences: {
          selected_model: selectedModel,
          context_enabled: contextEnabled,
          max_context_messages: maxContextMessages,
          console_max_entries: 50,
          active_tab: activeTab,
          show_history: false,
          // Drumcircle settings
          selected_drumcircle_models: selectedDrumcircleModels,
          drumcircle_task_type: drumcircleTaskType,
          drumcircle_rounds: drumcircleRounds,
          include_gemini: includeGemini,
        },
        llm_config_yaml: configYaml,
      });
    }, 2000); // 2 second delay
    return () => clearTimeout(timed);
  }, [config, apiKeys, selectedModel, contextEnabled, maxContextMessages, activeTab, configYaml, selectedDrumcircleModels, drumcircleTaskType, drumcircleRounds, includeGemini]);

  // Tab content now handled by extracted SecureTabRenderer with security checks

  return (
    <SecurityProvider remoteHost={remoteHost} config={config}>
      <SecurityWrapper useLoginPage={true}>
        <div className="app compact-ui">
          {/* 🂂 TAB NAVIGATION AND CONTENT WRAPPER */}
          <div className="main-panel">
            <div className="tab-navigation">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                <div style={{ display: 'flex', flexWrap: 'wrap' }}>
                  <RoleBasedTabs activeTab={activeTab} setActiveTab={setActiveTab} />
                </div>
              </div>
            </div>

            {/* 📄 TAB CONTENT */}
            <div className="tab-content">
              <ErrorBoundary component="SecureTabRenderer">
                <SecureTabRenderer
                  activeTab={activeTab}
                  selectedModel={selectedModel}
                  setSelectedModel={setSelectedModel}
                  availableModels={availableModels}
                  setAvailableModels={setAvailableModels}
                  contextEnabled={contextEnabled}
                  setContextEnabled={setContextEnabled}
                  maxContextMessages={maxContextMessages}
                  conversationHistory={conversationHistory}
                  messages={messages}
                  pendingMessages={pendingMessages} // 🚀 NEW: Pending messages with status
                  wsRef={wsRef}
                  config={config}
                  remoteHost={remoteHost}
                  connected={connected}
                  chatWithModel={chatWithModel}
                  toggleContext={toggleContext}
                  clearConversation={clearConversation}
                  handleTestOllamaModels={handleTestOllamaModels}
                  sendMessage={sendMessage}
                  updateConfig={updateConfig}
                  updateConfigDebounced={updateConfigDebounced}
                  startDrumcircle={startDrumcircle}
                  startConfigurableDrumcircle={startConfigurableDrumcircle}
                  configYaml={configYaml}
                  setConfigYaml={setConfigYaml}
                  llmProviders={llmProviders}
                  setLlmProviders={setLlmProviders}
                  setConfig={setConfig}
                  apiKeys={apiKeys}
                  setApiKeys={setApiKeys}
                  setRemoteHost={setRemoteHost}
                  drumcircleRoleMapping={drumcircleRoleMapping}
                  setDrumcircleRoleMapping={setDrumcircleRoleMapping}
                  activeThreads={activeThreads}
                  setActiveThreads={setActiveThreads}
                  selectedThread={selectedThread}
                  setSelectedThread={setSelectedThread}
                  startThreadedConversation={startThreadedConversation}
                  fetchActiveThreads={fetchActiveThreads}
                  viewThread={viewThread}
                  showHistory={showHistory}
                  setShowHistory={setShowHistory}
                  peers={peers}
                  drumcircleTemperature={drumcircleTemperature}
                  setDrumcircleTemperature={setDrumcircleTemperature}
                  drumcircleResponseLength={drumcircleResponseLength}
                  setDrumcircleResponseLength={setDrumcircleResponseLength}
                  drumcircleOutputFormat={drumcircleOutputFormat}
                  setDrumcircleOutputFormat={setDrumcircleOutputFormat}
                  drumcircleStreamingMode={drumcircleStreamingMode}
                  setDrumcircleStreamingMode={setDrumcircleStreamingMode}
                  drumcircleTimeout={drumcircleTimeout}
                  setDrumcircleTimeout={setDrumcircleTimeout}
                  drumcircleDebugMode={drumcircleDebugMode}
                  setDrumcircleDebugMode={setDrumcircleDebugMode}
                  drumcircleRetryFailures={drumcircleRetryFailures}
                  setDrumcircleRetryFailures={setDrumcircleRetryFailures}
                  drumcircleChaosMode={drumcircleChaosMode}
                  setDrumcircleChaosMode={setDrumcircleChaosMode}
                  consoleMaxEntries={consoleMaxEntries}
                  setConsoleMaxEntries={setConsoleMaxEntries}
                  editorRef={editorRef}
                  isRichText={isRichText}
                  setIsRichText={setIsRichText}
                  editorType={editorType}
                  setEditorType={setEditorType}
                  handlePaste={handlePaste}
                  handleInput={handleInput}
                  fetchConversationHistory={fetchConversationHistory}
                  setActiveTab={setActiveTab}
                  setConversationHistory={setConversationHistory}
                  updateApiKeys={updateApiKeys}
                  loadContinueConfigFromYaml={loadContinueConfigFromYaml}
                  threadPrompt={threadPrompt}
                  setThreadPrompt={setThreadPrompt}
                  selectedThreadModels={selectedThreadModels}
                  setSelectedThreadModels={setSelectedThreadModels}
                  maxThreadRounds={maxThreadRounds}
                  setMaxThreadRounds={setMaxThreadRounds}
                />
              </ErrorBoundary>
            </div>

            {/* TODO: Add status bar, WebSocket status, etc. */}
          </div> {/* Closing main-panel */}
        </div>
      </SecurityWrapper>
    </SecurityProvider>
  );
};

// 🛡️ SECURITY-WRAPPED APP EXPORT
const SecureApp = () => {
  return <App />;
};

export default SecureApp;
