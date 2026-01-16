import { useState } from 'react'
import WebcamCapture from '../components/WebcamCapture'
import VoiceRecorder from '../components/VoiceRecorder'

const MAX_FILE_SIZE = 10 * 1024 * 1024 // 10MB

function VerifyPage() {
    const [did, setDid] = useState('')
    const [files, setFiles] = useState({
        face: null,
        voice: null,
        idDoc: null
    })
    const [errors, setErrors] = useState({})
    const [verifying, setVerifying] = useState(false)
    const [result, setResult] = useState(null)

    // Modal states
    const [showWebcam, setShowWebcam] = useState(false)
    const [showVoiceRecorder, setShowVoiceRecorder] = useState(false)

    const validateFile = (file, type, required = true) => {
        if (!file) return required ? 'File is required' : null

        if (file.size > MAX_FILE_SIZE) {
            return 'File size must be under 10MB'
        }

        if (type === 'image' && !['image/jpeg', 'image/jpg'].includes(file.type)) {
            return 'File must be JPEG format'
        }

        if (type === 'audio') {
            const validTypes = ['audio/wav', 'audio/wave', 'audio/x-wav', 'audio/webm', 'audio/mpeg', 'audio/mp4', 'audio/ogg']
            if (!validTypes.includes(file.type)) {
                return 'Invalid audio format'
            }
        }

        return null
    }

    const handleFileChange = (field, type, required = true) => (e) => {
        const file = e.target.files[0]
        if (file) {
            const error = validateFile(file, type, required)
            setErrors(prev => ({ ...prev, [field]: error }))
            setFiles(prev => ({ ...prev, [field]: error ? null : file }))
        }
    }

    const handleWebcamCapture = (file) => {
        setFiles(prev => ({ ...prev, face: file }))
        setErrors(prev => ({ ...prev, face: null }))
        setShowWebcam(false)
    }

    const handleVoiceCapture = (file) => {
        setFiles(prev => ({ ...prev, voice: file }))
        setErrors(prev => ({ ...prev, voice: null }))
        setShowVoiceRecorder(false)
    }

    const handleSubmit = async (e) => {
        e.preventDefault()

        const newErrors = {
            did: !did ? 'Short code or DID is required' : null,
            face: validateFile(files.face, 'image', true),
            voice: validateFile(files.voice, 'audio', true),
            idDoc: validateFile(files.idDoc, 'image', false)
        }

        setErrors(newErrors)

        if (newErrors.did || newErrors.face || newErrors.voice) {
            return
        }

        setVerifying(true)
        setResult(null)

        const formData = new FormData()
        formData.append('did', did)
        formData.append('face', files.face)
        formData.append('voice', files.voice)
        if (files.idDoc) {
            formData.append('id_doc', files.idDoc)
        }

        try {
            const response = await fetch('/api/verify', {
                method: 'POST',
                body: formData
            })

            const data = await response.json()

            if (response.ok) {
                setResult({ success: true, data })
            } else {
                setResult({ success: false, error: data.detail || 'Verification failed' })
            }
        } catch (error) {
            setResult({ success: false, error: 'Network error. Please try again.' })
        } finally {
            setVerifying(false)
        }
    }

    const getConfidenceBadgeClass = (level) => {
        switch (level) {
            case 'VERY_HIGH': return 'badge badge-success'
            case 'HIGH': return 'badge badge-info'
            case 'MEDIUM': return 'badge badge-warning'
            default: return 'badge badge-error'
        }
    }

    const formatScore = (score) => {
        return (score * 100).toFixed(1) + '%'
    }

    const getScoreColor = (score) => {
        if (score >= 0.8) return 'var(--success-light)'
        if (score >= 0.6) return 'var(--info-light)'
        if (score >= 0.4) return 'var(--warning-light)'
        return 'var(--error-light)'
    }

    return (
        <div className="page">
            {/* Hero Header */}
            <div className="page-header">
                <div style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    marginBottom: '1rem',
                    padding: '0.5rem 1rem',
                    background: 'rgba(6, 182, 212, 0.1)',
                    borderRadius: 'var(--radius-full)',
                    border: '1px solid rgba(6, 182, 212, 0.2)'
                }}>
                    <span style={{ fontSize: '1.25rem' }}>🔍</span>
                    <span style={{
                        fontSize: 'var(--font-size-xs)',
                        color: '#22d3ee',
                        fontWeight: '600',
                        letterSpacing: '1px',
                        textTransform: 'uppercase'
                    }}>Authenticate</span>
                </div>
                <h1 className="page-title">Verify Identity</h1>
                <p className="page-subtitle">Authenticate using decentralized biometric verification powered by blockchain</p>
            </div>

            {/* Verification Flow Indicator */}
            <div className="card" style={{
                marginBottom: '1.5rem',
                background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.05) 0%, rgba(99, 102, 241, 0.05) 100%)',
                border: '1px solid rgba(6, 182, 212, 0.2)',
                padding: 'var(--spacing-lg)'
            }}>
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexWrap: 'wrap',
                    gap: '0.5rem 1rem',
                    fontSize: 'var(--font-size-xs)'
                }}>
                    {[
                        { icon: '⛓️', label: 'Blockchain' },
                        { icon: '📦', label: 'IPFS' },
                        { icon: '🔓', label: 'Decrypt' },
                        { icon: '🔄', label: 'Compare' },
                        { icon: '✅', label: 'Proof' }
                    ].map((step, index, arr) => (
                        <div key={step.label} style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.75rem'
                        }}>
                            <div style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.5rem',
                                padding: '0.5rem 0.875rem',
                                background: 'rgba(255, 255, 255, 0.03)',
                                borderRadius: 'var(--radius-full)',
                                border: '1px solid var(--border-color)',
                                whiteSpace: 'nowrap'
                            }}>
                                <span>{step.icon}</span>
                                <span style={{ color: 'var(--text-secondary)' }}>{step.label}</span>
                            </div>
                            {index < arr.length - 1 && (
                                <span style={{ color: 'var(--text-muted)' }}>→</span>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* Main Verification Card */}
            <div className="card">
                <form onSubmit={handleSubmit}>
                    {/* Identity Input */}
                    <div className="form-group">
                        <label className="form-label" htmlFor="did">
                            <span>🔑</span>
                            <span>Identity Code or DID</span>
                        </label>
                        <input
                            id="did"
                            type="text"
                            className="form-input"
                            placeholder="Enter short code (e.g., MZXW6YTB) or full DID"
                            value={did}
                            onChange={(e) => setDid(e.target.value)}
                            disabled={verifying}
                            style={{
                                textTransform: did.startsWith('did:') ? 'none' : 'uppercase',
                                fontFamily: 'var(--font-mono)',
                                letterSpacing: '0.15rem',
                                fontSize: 'var(--font-size-lg)',
                                textAlign: 'center'
                            }}
                        />
                        <div style={{
                            fontSize: 'var(--font-size-xs)',
                            color: 'var(--text-muted)',
                            marginTop: '0.5rem',
                            textAlign: 'center'
                        }}>
                            Enter your <strong style={{ color: 'var(--primary-light)' }}>8-character short code</strong> or full DID
                        </div>
                        {errors.did && <p className="status status-error" style={{ marginTop: '0.5rem', padding: '0.75rem' }}>{errors.did}</p>}
                    </div>

                    {/* Divider */}
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '1rem',
                        margin: '2rem 0'
                    }}>
                        <div style={{ flex: 1, height: '1px', background: 'var(--border-color)' }}></div>
                        <span style={{
                            color: 'var(--text-muted)',
                            fontSize: 'var(--font-size-xs)',
                            textTransform: 'uppercase',
                            letterSpacing: '1px'
                        }}>Biometric Verification</span>
                        <div style={{ flex: 1, height: '1px', background: 'var(--border-color)' }}></div>
                    </div>

                    {/* Face Upload */}
                    <div className="form-group">
                        <label className="form-label">
                            <span>📸</span>
                            <span>Live Face Image</span>
                            <span className="badge badge-info" style={{ marginLeft: 'auto', fontSize: '0.6rem' }}>Required</span>
                        </label>
                        <div className={`file-upload ${files.face ? 'has-file' : ''} ${errors.face ? 'error' : ''}`}>
                            <input
                                type="file"
                                className="file-upload-input"
                                accept="image/jpeg,image/jpg"
                                onChange={handleFileChange('face', 'image', true)}
                                disabled={verifying}
                            />
                            <div className="file-upload-icon">📸</div>
                            <div className="file-upload-text">
                                {files.face ? files.face.name : 'Click to upload live face photo'}
                            </div>
                            <div className="file-upload-hint">JPEG format • Max 10MB</div>
                        </div>
                        <div className="capture-options">
                            <button
                                type="button"
                                className={`capture-btn ${files.face ? 'active' : ''}`}
                                onClick={() => setShowWebcam(true)}
                                disabled={verifying}
                            >
                                📷 Use Camera
                            </button>
                        </div>
                        {errors.face && <p className="status status-error" style={{ marginTop: '0.5rem', padding: '0.75rem' }}>{errors.face}</p>}
                    </div>

                    {/* Voice Upload */}
                    <div className="form-group">
                        <label className="form-label">
                            <span>🎤</span>
                            <span>Live Voice Sample</span>
                            <span className="badge badge-info" style={{ marginLeft: 'auto', fontSize: '0.6rem' }}>Required</span>
                        </label>
                        <div className={`file-upload ${files.voice ? 'has-file' : ''} ${errors.voice ? 'error' : ''}`}>
                            <input
                                type="file"
                                className="file-upload-input"
                                accept="audio/*"
                                onChange={handleFileChange('voice', 'audio', true)}
                                disabled={verifying}
                            />
                            <div className="file-upload-icon">🎤</div>
                            <div className="file-upload-text">
                                {files.voice ? files.voice.name : 'Click to upload live voice recording'}
                            </div>
                            <div className="file-upload-hint">Any audio format • Max 10MB</div>
                        </div>
                        <div className="capture-options">
                            <button
                                type="button"
                                className={`capture-btn ${files.voice ? 'active' : ''}`}
                                onClick={() => setShowVoiceRecorder(true)}
                                disabled={verifying}
                            >
                                🎙️ Record Voice
                            </button>
                        </div>
                        {errors.voice && <p className="status status-error" style={{ marginTop: '0.5rem', padding: '0.75rem' }}>{errors.voice}</p>}
                    </div>

                    {/* ID Document Upload (Optional) */}
                    <div className="form-group">
                        <label className="form-label">
                            <span>🪪</span>
                            <span>ID Document</span>
                            <span className="badge" style={{
                                marginLeft: 'auto',
                                fontSize: '0.6rem',
                                background: 'rgba(255, 255, 255, 0.05)',
                                color: 'var(--text-muted)',
                                border: '1px solid var(--border-color)'
                            }}>Optional</span>
                        </label>
                        <div className={`file-upload ${files.idDoc ? 'has-file' : ''} ${errors.idDoc ? 'error' : ''}`}>
                            <input
                                type="file"
                                className="file-upload-input"
                                accept="image/jpeg,image/jpg"
                                onChange={handleFileChange('idDoc', 'image', false)}
                                disabled={verifying}
                            />
                            <div className="file-upload-icon">🪪</div>
                            <div className="file-upload-text">
                                {files.idDoc ? files.idDoc.name : 'Upload for enhanced OCR verification'}
                            </div>
                            <div className="file-upload-hint">JPEG format • Max 10MB</div>
                        </div>
                        {errors.idDoc && <p className="status status-error" style={{ marginTop: '0.5rem', padding: '0.75rem' }}>{errors.idDoc}</p>}
                    </div>

                    {/* Submit Button */}
                    <button
                        type="submit"
                        className="btn btn-primary btn-full"
                        disabled={verifying}
                        style={{ marginTop: '0.5rem' }}
                    >
                        {verifying ? (
                            <>
                                <span className="loader"></span>
                                Fetching from IPFS & Verifying...
                            </>
                        ) : (
                            <>
                                <span style={{ fontSize: '1.1rem' }}>🔍</span>
                                Verify Identity
                            </>
                        )}
                    </button>
                </form>

                {/* Result Display */}
                {result && (
                    <div className={`status ${result.success && result.data.verified ? 'status-success' : 'status-error'}`}>
                        {result.success ? (
                            <div>
                                {/* Header with Icon and Badge */}
                                <div style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '1rem',
                                    marginBottom: '1.5rem'
                                }}>
                                    <div style={{
                                        width: '56px',
                                        height: '56px',
                                        borderRadius: '50%',
                                        background: result.data.verified
                                            ? 'linear-gradient(135deg, var(--success) 0%, #22d3ee 100%)'
                                            : 'linear-gradient(135deg, var(--error) 0%, #f97316 100%)',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        fontSize: '1.75rem',
                                        boxShadow: result.data.verified
                                            ? '0 4px 20px rgba(34, 197, 94, 0.4)'
                                            : '0 4px 20px rgba(239, 68, 68, 0.4)'
                                    }}>
                                        {result.data.verified ? '✓' : '✗'}
                                    </div>
                                    <div style={{ flex: 1 }}>
                                        <p style={{ fontWeight: 700, fontSize: '1.25rem', margin: 0 }}>
                                            {result.data.verified ? 'Identity Verified' : 'Verification Failed'}
                                        </p>
                                        <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-muted)', margin: 0 }}>
                                            {result.data.verified ? 'Biometric match confirmed' : 'Biometrics do not match'}
                                        </p>
                                    </div>
                                    <span className={getConfidenceBadgeClass(result.data.confidence_level)}>
                                        {result.data.confidence_level?.replace('_', ' ')}
                                    </span>
                                </div>

                                {/* IPFS CID */}
                                <div style={{ marginBottom: '1.5rem' }}>
                                    <div style={{
                                        color: 'var(--text-muted)',
                                        fontSize: '0.65rem',
                                        marginBottom: '0.5rem',
                                        textTransform: 'uppercase',
                                        letterSpacing: '0.5px'
                                    }}>IPFS CID Retrieved from Blockchain</div>
                                    <div className="did-display" style={{ fontSize: '0.7rem' }}>
                                        {result.data.metadata_cid}
                                    </div>
                                </div>

                                {/* Score Display */}
                                {/* Final Score - Prominent */}
                                <div style={{
                                    textAlign: 'center',
                                    padding: '1.5rem',
                                    background: 'rgba(99, 102, 241, 0.1)',
                                    border: '1px solid rgba(99, 102, 241, 0.3)',
                                    borderRadius: 'var(--radius-lg)',
                                    marginBottom: '1rem'
                                }}>
                                    <div style={{
                                        fontSize: 'var(--font-size-4xl)',
                                        fontWeight: 700,
                                        background: 'var(--primary-gradient)',
                                        WebkitBackgroundClip: 'text',
                                        WebkitTextFillColor: getScoreColor(result.data.final_score)
                                    }}>{formatScore(result.data.final_score)}</div>
                                    <div style={{
                                        fontSize: 'var(--font-size-xs)',
                                        color: 'var(--text-muted)',
                                        marginTop: '0.5rem',
                                        fontWeight: 500,
                                        textTransform: 'uppercase',
                                        letterSpacing: '1px'
                                    }}>Final Score</div>
                                </div>

                                {/* Individual Scores */}
                                <div style={{
                                    display: 'grid',
                                    gridTemplateColumns: 'repeat(3, 1fr)',
                                    gap: 'var(--spacing-md)'
                                }}>
                                    <div className="score-item">
                                        <div className="score-value">{formatScore(result.data.face_score)}</div>
                                        <div className="score-label">Face (40%)</div>
                                    </div>
                                    <div className="score-item">
                                        <div className="score-value">{formatScore(result.data.voice_score)}</div>
                                        <div className="score-label">Voice (35%)</div>
                                    </div>
                                    <div className="score-item">
                                        <div className="score-value">{formatScore(result.data.doc_score)}</div>
                                        <div className="score-label">Doc (25%)</div>
                                    </div>
                                </div>

                                {/* Detailed Document Scores */}
                                {(result.data.doc_text_score !== undefined || result.data.doc_face_score !== undefined) && (
                                    <div className="score-display" style={{ marginTop: '0.75rem' }}>
                                        <div className="score-item" style={{
                                            background: 'rgba(139, 92, 246, 0.1)',
                                            borderColor: 'rgba(139, 92, 246, 0.3)'
                                        }}>
                                            <div className="score-value">{formatScore(result.data.doc_text_score || 0)}</div>
                                            <div className="score-label">OCR Text</div>
                                        </div>
                                        <div className="score-item" style={{
                                            background: 'rgba(139, 92, 246, 0.1)',
                                            borderColor: 'rgba(139, 92, 246, 0.3)'
                                        }}>
                                            <div className="score-value">{formatScore(result.data.doc_face_score || 0)}</div>
                                            <div className="score-label">Doc Face</div>
                                        </div>
                                    </div>
                                )}

                                {/* Transaction Hash */}
                                {result.data.tx_hash && (
                                    <div style={{
                                        marginTop: '1.5rem',
                                        padding: '1rem',
                                        background: 'rgba(34, 197, 94, 0.08)',
                                        borderRadius: 'var(--radius-md)',
                                        border: '1px solid rgba(34, 197, 94, 0.2)'
                                    }}>
                                        <div style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '0.5rem',
                                            marginBottom: '0.5rem'
                                        }}>
                                            <span style={{ fontSize: '1rem' }}>✅</span>
                                            <span style={{
                                                fontSize: '0.65rem',
                                                color: 'var(--success-light)',
                                                textTransform: 'uppercase',
                                                letterSpacing: '0.5px',
                                                fontWeight: '600'
                                            }}>Verification Proof on Blockchain</span>
                                        </div>
                                        <p className="tx-hash">
                                            <a href={`https://sepolia.etherscan.io/tx/${result.data.tx_hash}`} target="_blank" rel="noopener noreferrer">
                                                {result.data.tx_hash}
                                            </a>
                                        </p>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                                <span style={{ fontSize: '1.5rem' }}>❌</span>
                                <p style={{ margin: 0 }}>{result.error}</p>
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Webcam Modal */}
            {showWebcam && (
                <WebcamCapture
                    onCapture={handleWebcamCapture}
                    onClose={() => setShowWebcam(false)}
                />
            )}

            {/* Voice Recorder Modal */}
            {showVoiceRecorder && (
                <VoiceRecorder
                    onCapture={handleVoiceCapture}
                    onClose={() => setShowVoiceRecorder(false)}
                />
            )}
        </div>
    )
}

export default VerifyPage
