# Documentation

Welcome to the Shared Ollama Service documentation. This directory contains comprehensive documentation for users, developers, and operators.

## 📚 Documentation Index

### Getting Started

- **[Configuration Guide](CONFIGURATION.md)** - Complete configuration reference with environment variables, validation, and examples
- **[Integration Guide](INTEGRATION_GUIDE.md)** - How to integrate the service into your projects
- **[Migration Guide](MIGRATION_GUIDE.md)** - Migrating from individual Ollama instances to the shared service
- **[Client Guide](CLIENT_GUIDE.md)** - Quick start examples for curl, Python, TypeScript, and Go

### API Documentation

- **[API Reference](API_REFERENCE.md)** - Complete API documentation for all client methods and REST endpoints
- **[OpenAPI Specification](openapi.yaml)** - Complete API specification (OpenAPI 3.1.0)
- **[VLM Guide](VLM_GUIDE.md)** - Complete vision-language model guide with examples
- **[POML Guide](POML_GUIDE.md)** - Prompt Orchestration Markup Language support
- **[LiteLLM Guide](LITELLM_GUIDE.md)** - LiteLLM integration guide

### Architecture & Design

- **[Architecture](ARCHITECTURE.md)** - System architecture, component structure, and design decisions
- **[Scaling & Load Testing](SCALING_AND_LOAD_TESTING.md)** - Performance tuning, load testing, and scaling strategies

### Operations & Maintenance

- **[Operations Guide](OPERATIONS.md)** - Service operations, warm-up, and pre-loading
- **[Monitoring Guide](MONITORING.md)** - Monitoring, metrics, and observability
- **[Resource Management](RESOURCE_MANAGEMENT.md)** - Memory usage and performance tuning
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions

### Development

- **[Development Guide](DEVELOPMENT.md)** - Development setup, testing, code style, and contribution guidelines
- **[Model Storage](MODEL_STORAGE.md)** - Where models are stored and how to manage them

### Project History

- **[Changelog](CHANGELOG.md)** - Version history and release notes
- **[Stability Plan](STABILITY_PLAN.md)** - Hardening roadmap and stabilization plan

## 🚀 Quick Start

1. **Install**: See main [README.md](../README.md)
2. **Configure**: Edit `config/models.yaml` if you need to change which models load for each hardware tier
3. **Start**: Run `./scripts/core/start.sh`
4. **Integrate**: See [Integration Guide](INTEGRATION_GUIDE.md)

## 📖 Documentation Structure

```
docs/
├── README.md                    # This file - documentation index
├── CONFIGURATION.md             # Configuration guide
├── ARCHITECTURE.md              # System architecture
├── API_REFERENCE.md             # API documentation
├── INTEGRATION_GUIDE.md         # Integration instructions
├── MIGRATION_GUIDE.md           # Migration guide
├── CLIENT_GUIDE.md             # Client quickstart examples
├── VLM_GUIDE.md                # Vision-language model guide
├── POML_GUIDE.md               # POML support guide
├── LITELLM_GUIDE.md            # LiteLLM integration guide
├── OPERATIONS.md                # Service operations
├── MONITORING.md                # Monitoring and observability
├── RESOURCE_MANAGEMENT.md       # Memory and performance
├── TROUBLESHOOTING.md           # Troubleshooting guide
├── DEVELOPMENT.md               # Development guide
├── SCALING_AND_LOAD_TESTING.md  # Performance guide
├── MODEL_STORAGE.md             # Model storage information
├── CHANGELOG.md                 # Version history
├── STABILITY_PLAN.md            # Stability roadmap
├── openapi.yaml                 # OpenAPI specification
└── archive/                     # Historical documentation
```

## 🔍 Finding Information

- **New to the project?** Start with [Integration Guide](INTEGRATION_GUIDE.md) or [Client Guide](CLIENT_GUIDE.md)
- **Using VLM?** See [VLM Guide](VLM_GUIDE.md)
- **Configuring the service?** See [Configuration Guide](CONFIGURATION.md)
- **Understanding the system?** Read [Architecture](ARCHITECTURE.md)
- **Troubleshooting issues?** Check [Troubleshooting Guide](TROUBLESHOOTING.md)
- **Monitoring the service?** See [Monitoring Guide](MONITORING.md)
- **Developing features?** Check [Development Guide](DEVELOPMENT.md)
- **API questions?** See [API Reference](API_REFERENCE.md)

## 📝 Contributing

When adding or updating documentation:

1. Follow the existing structure and style
2. Update this index if adding new documents
3. Keep documentation up-to-date with code changes
4. Use clear, concise language
5. Include examples where helpful

## 🔗 External Links

- **Main README**: [../README.md](../README.md)
- **Project Repository**: See main README for repository URL
- **Ollama Documentation**: https://ollama.ai/docs
