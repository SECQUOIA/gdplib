# MCP Server Configuration for GDPlib

## Model Context Protocol Setup
This configuration enables intelligent code assistance for GDP (Generalized Disjunctive Programming) development.

### Server Configuration
```json
{
  "mcpServers": {
    "pyomo-assistant": {
      "command": "python",
      "args": ["-m", "mcp_server_pyomo"],
      "env": {
        "PROJECT_TYPE": "gdp-library",
        "PYOMO_VERSION": ">=5.6.1"
      }
    },
    "math-solver": {
      "command": "python", 
      "args": ["-m", "mcp_server_math"],
      "env": {
        "DOMAIN": "optimization",
        "SOLVER_TYPES": "gdp,minlp,milp"
      }
    }
  }
}
```

### Context Providers
- **Pyomo Model Context**: Understands GDP constructs, constraints, and variables
- **Mathematical Formulation**: Assists with constraint formulation and objective functions
- **Documentation Helper**: Generates appropriate docstrings and README content
- **Testing Assistant**: Creates relevant test cases for optimization models

### Usage Instructions
1. Install required MCP servers: `pip install mcp-server-pyomo mcp-server-math`
2. Configure your IDE/editor to use these MCP servers
3. The servers will provide domain-specific assistance for GDP model development

### Features Enabled
- Smart suggestions for Pyomo constructs
- Mathematical notation assistance
- Model validation helpers
- Performance optimization tips
- Documentation generation