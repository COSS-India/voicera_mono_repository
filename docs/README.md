# VoiceERA Documentation

Welcome to the official VoiceERA documentation!

This documentation is built using [MkDocs](https://www.mkdocs.org/) with the [Material theme](https://squidfunk.github.io/mkdocs-material/).

## Quick Links

- **[Getting Started](getting-started/installation.md)** - Installation and quick start
- **[Architecture](architecture/overview.md)** - System design and architecture
- **[API Documentation](api/rest-api.md)** - REST and WebSocket APIs
- **[Deployment](deployment/docker.md)** - Docker and production deployment
- **[Contributing](development/contributing.md)** - How to contribute

## Building Documentation Locally

### Prerequisites

```bash
pip install mkdocs mkdocs-material
```

### Build and Serve

```bash
# Serve locally with auto-reload
mkdocs serve

# Open http://localhost:8000 in your browser
```

### Build Static Site

```bash
mkdocs build

# Output in ./site directory
```

## Documentation Structure

```
docs/
├── index.md                          # Documentation homepage
├── getting-started/
│   ├── installation.md               # Installation guide
│   ├── quickstart.md                 # 5-minute quick start
│   └── configuration.md              # Configuration reference
├── architecture/
│   ├── overview.md                   # System architecture
│   ├── system-design.md              # Component design
│   └── data-flow.md                  # Data flow through system
├── services/
│   ├── backend.md                    # Backend API service
│   ├── voice-server.md               # Voice processing server
│   ├── frontend.md                   # Frontend web app
│   ├── ai4bharat-stt.md              # STT service
│   └── ai4bharat-tts.md              # TTS service
├── api/
│   ├── rest-api.md                   # REST API reference
│   ├── websocket-api.md              # WebSocket API
│   └── endpoints.md                  # Quick endpoint reference
├── deployment/
│   ├── docker.md                     # Docker deployment
│   ├── production.md                 # Production deployment
│   └── environment.md                # Environment variables
├── development/
│   ├── local-setup.md                # Local development setup
│   ├── contributing.md               # Contribution guidelines
│   └── testing.md                    # Testing guide
└── troubleshooting.md                # Troubleshooting guide
```

## Key Sections

### Getting Started
- **Installation**: Platform-specific installation instructions
- **Quick Start**: Get VoiceERA running in 5 minutes
- **Configuration**: Complete environment variable reference

### Architecture
- **Overview**: High-level system architecture (C4 Model)
- **System Design**: Detailed component design and implementations
- **Data Flow**: How data moves through the system

### Services
- **Backend API**: FastAPI backend documentation
- **Voice Server**: Real-time voice processing with Pipecat
- **Frontend**: Next.js web dashboard
- **AI4Bharat STT/TTS**: Indic language speech services

### API
- **REST API**: HTTP API endpoints and examples
- **WebSocket**: Real-time voice communication
- **Endpoints**: Quick reference table of all endpoints

### Deployment
- **Docker**: Local Docker and Docker Compose setup
- **Production**: Production-grade deployment guide
- **Environment**: All environment variables with descriptions

### Development
- **Local Setup**: Development environment setup
- **Contributing**: How to contribute to VoiceERA
- **Testing**: Testing guidelines and examples

### Troubleshooting
- Service startup issues
- Common errors and solutions
- Performance debugging
- Getting help

## Documentation Guidelines

### Writing Docs

1. **Use clear, concise language**
   - Avoid jargon where possible
   - Define technical terms on first use
   - Keep sentences short

2. **Include examples**
   - Code examples for developers
   - Command examples for operators
   - Configuration examples
   - Real-world use cases

3. **Use proper formatting**
   - Headers for sections (# Main, ## Sub, ### Details)
   - Code blocks with language tags
   - Tables for structured data
   - Lists for sequential steps
   - Blockquotes for important notes

4. **Add cross-references**
   - Link to related documentation
   - Link to external resources
   - Link to code in repository

5. **Keep current**
   - Update when features change
   - Remove deprecated information
   - Add new features immediately

### Markdown Formatting

```markdown
# Main Header
## Subheader
### Sub-subheader

**Bold text**
*Italic text*
`code`

```python
# Code block
def hello():
    print("Hello")
```

> Important note or quote

- List item 1
- List item 2
  - Nested item

| Header 1 | Header 2 |
|----------|----------|
| Data 1   | Data 2   |

[Link text](url)
```

### Using Admonitions

```markdown
!!! note "Title"
    This is a note

!!! warning "Title"
    This is a warning

!!! danger "Title"
    This is a danger warning

!!! success "Title"
    This is a success message
```

Renders as:

!!! note "Note"
    This is a note

!!! warning "Warning"
    This is a warning

!!! danger "Danger"
    This is a danger warning

## Contributing to Docs

### Small Changes

1. Fork the repository
2. Edit `.md` files in `docs/` folder
3. Submit a pull request

### Large Changes

1. Create an issue to discuss changes
2. Fork the repository
3. Create a feature branch: `git checkout -b docs/feature-name`
4. Make changes
5. Build locally to verify: `mkdocs serve`
6. Submit a pull request

### Checklist

- [ ] Content is accurate and up-to-date
- [ ] Examples are tested and working
- [ ] Cross-references are correct
- [ ] Formatting is consistent
- [ ] No broken links
- [ ] No spelling errors

## Building & Deploying

### Local Preview

```bash
cd docs
mkdocs serve
# Open http://localhost:8000
```

### Deploy to GitHub Pages

```bash
mkdocs gh-deploy
```

### Deploy to Custom Host

```bash
mkdocs build
# Upload ./site to your web server
```

## Documentation Support

- **Questions**: Open a GitHub issue
- **Corrections**: Submit a pull request
- **Feedback**: Start a discussion on GitHub
- **Major Updates**: Check the roadmap

## Tech Stack

- **MkDocs**: Static site generator
- **Material Theme**: Professional theme with search
- **Python Markdown**: Markdown processor
- **Mermaid**: Diagrams (if needed)
- **KaTeX**: Math equations (if needed)

## License

Documentation is under the same license as VoiceERA.

---

Made with ❤️ by the VoiceERA Team
