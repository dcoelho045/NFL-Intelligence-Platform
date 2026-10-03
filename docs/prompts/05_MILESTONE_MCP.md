# Milestone 5 — Model Context Protocol (MCP)

## Objective

Build an NFL MCP server that exposes the existing analytics capabilities through the Model Context Protocol.

## Architecture

```text
NFL Analytics Service
        |
        v
     MCP Server
        |
        +--> get_player_stats
        +--> get_team_stats
        +--> get_matchup
        +--> get_injury
        +--> predict_game
        +--> compare_players
```

## Important Architecture Rule

MCP is an interface layer.

Do NOT duplicate business logic inside MCP tool implementations.

MCP tools should call the existing analytics/model services.

## MCP Capabilities

Implement appropriate MCP:

### Tools

Expose useful analytics operations.

### Resources

Where useful, expose resources such as:

```text
player://...
team://...
game://...
season://...
```

Only implement resources that have a clear use case.

### Prompts

Consider reusable prompts such as:

```text
analyze_matchup
fantasy_start_sit
game_prediction
```

## Security

Do not expose arbitrary filesystem/database access.

Validate tool arguments.

Do not allow user-provided input to become arbitrary code or queries.

## Testing

Test:

- MCP initialization
- tool discovery
- tool arguments
- tool execution
- error handling
- response schemas

## Documentation

Create:

`docs/mcp-design.md`

Explain:

- what MCP is
- why it is useful here
- why MCP is used at this boundary
- how MCP differs from a normal REST API
- how tools map to the analytics service

## Acceptance Criteria

- [x] MCP server starts
- [x] Tools are discoverable
- [x] Tools execute successfully
- [x] Business logic remains outside MCP
- [x] Tests pass
- [x] MCP architecture is documented

## Copilot Instructions

Use the current official MCP Python SDK documentation appropriate to the installed version.

Do not invent MCP protocol behavior.

If the SDK/API has changed, inspect the installed package and current official documentation before implementation.

Do not implement A2A yet.
