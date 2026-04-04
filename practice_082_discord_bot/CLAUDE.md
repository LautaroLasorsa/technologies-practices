# Discord Bot Engineering: Gateway, Interactions & UI Components

## Technologies
discord.py 2.4+, python-dotenv

## Stack
Python 3.12

## Theoretical Context

### What Discord Bots Are

A Discord bot is an application that connects to Discord's **Gateway** — a persistent WebSocket connection — and responds to events in real time. Unlike HTTP APIs where your server receives requests, the Gateway is a **push model**: Discord pushes events (messages, reactions, member joins) to your bot over the WebSocket, and your bot registers async handlers to process them. This is conceptually similar to a Kafka consumer: you subscribe to event streams and process them as they arrive.

The bot runs as a **separate Discord user** (a "bot account") created through the [Developer Portal](https://discord.com/developers/applications). You invite this bot to a server (guild) using an OAuth2 URL. Users interact with the bot from their normal accounts — the bot sees their actions as events.

### The Gateway & Intents Model

Discord's Gateway sends a high volume of events. To avoid overwhelming bots with data they don't need, Discord introduced **Intents** — a bitmask that acts as a subscription filter. When your bot connects, it declares which event categories it wants:

| Intent | Events Enabled | Privileged? |
|--------|---------------|-------------|
| `guilds` | Guild create/update/delete, channel CRUD | No |
| `guild_members` | Member join/leave/update | **Yes** |
| `guild_messages` | Messages in guild channels | No |
| `message_content` | Actual text content of messages | **Yes** |
| `guild_reactions` | Reaction add/remove | No |

**Privileged intents** (`members`, `presences`, `message_content`) require explicit opt-in in the Developer Portal. Without enabling `message_content`, your bot receives message events but the `content` field is empty — this is the **#1 beginner error**.

In code, intents work like this:
```python
intents = discord.Intents.default()       # All non-privileged intents
intents.message_content = True            # Opt into message content
intents.members = True                    # Opt into member events
```

Think of it like Kafka consumer groups filtering topics: you only receive what you subscribe to, reducing bandwidth and processing overhead.

### Slash Commands vs Prefix Commands

Historically, bots used **prefix commands** (`!kick @user`). Discord's modern approach uses **Slash Commands** (Application Commands) — first-class UI elements with:

- **Auto-complete** in the Discord client
- **Typed parameters** (string, integer, user, channel, role) with validation
- **Descriptions** shown in the UI
- **Permission integration** — Discord handles permission checks before the command reaches your bot

The key classes:

- **`app_commands.CommandTree`**: A registry that holds all your slash commands. Attached to the bot, it manages command state and syncing.
- **`@app_commands.command()`**: Decorator to define a slash command. Parameters are type-hinted and automatically converted by Discord.
- **`discord.Interaction`**: The request/response object for slash commands. You **must** respond within 3 seconds (or defer).

**Guild sync vs Global sync**: Commands synced to a specific guild appear instantly. Global commands take up to **1 hour** to propagate across all servers. During development, always use guild sync:
```python
# Instant — guild-specific
await tree.sync(guild=discord.Object(id=GUILD_ID))

# Slow — up to 1 hour propagation
await tree.sync()
```

### Cogs: Modular Command Groups

A **Cog** is discord.py's implementation of the **Plugin/Extension pattern**. Each Cog is a Python class (subclassing `commands.Cog`) that groups related commands, event listeners, and state into a self-contained module:

- Commands defined inside the Cog are automatically registered when the Cog is loaded
- Each Cog lives in its own file (an "extension") with an `async def setup(bot)` entry point
- Extensions are loaded dynamically via `bot.load_extension("cogs.my_cog")`
- Cogs can be loaded, unloaded, and reloaded at runtime — enabling hot-reloading during development

This pattern enforces **separation of concerns**: moderation commands in one Cog, info commands in another, games in a third. Each Cog only knows about its own domain.

### Interactions: Request/Response Flow

When a user triggers a slash command, button press, or modal submit:

1. Discord sends an **Interaction** to your bot via the Gateway
2. Your bot has **3 seconds** to respond (or defer with `interaction.response.defer()`)
3. Response options:
   - `send_message()` — text/embed reply
   - `send_modal()` — open a modal dialog
   - `defer()` — acknowledge, send followup later
4. After the initial response, use `interaction.followup.send()` for additional messages

Each interaction can only be responded to **once** (the initial response). Attempting a second `response.send_message()` raises `InteractionResponded`.

### Views & UI Components

**Views** are stateful containers for interactive UI components (buttons, select menus). A `discord.ui.View` is a Python object that:

- Contains up to 25 components arranged in up to 5 action rows
- Has a configurable **timeout** (default 180s) — after which buttons stop working
- Maintains state (e.g., vote counts) in memory
- Receives callbacks when users interact with its components

**Modals** (`discord.ui.Modal`) are popup forms with up to 5 `TextInput` fields. They can only be triggered from an interaction (slash command or button press) — you cannot send a modal unprompted.

Key lifecycle: View is created -> attached to a message via `send_message(view=view)` -> user clicks button -> `button.callback()` fires -> view state updates -> timeout eventually disables the view.

### The OAuth2 Bot Invite Flow

To add a bot to a server:

1. Create an **Application** in the Developer Portal
2. Create a **Bot user** under the application
3. Generate an **OAuth2 URL** with:
   - **Scopes**: `bot` (for Gateway access) + `applications.commands` (for slash commands)
   - **Permissions**: Select what the bot can do (send messages, manage roles, kick members, etc.)
4. The resulting URL looks like: `https://discord.com/oauth2/authorize?client_id=APP_ID&scope=bot+applications.commands&permissions=PERMISSION_INT`
5. Opening this URL lets you select a server and authorize the bot

### Rate Limits

Discord enforces **per-route rate limits**. Each API endpoint has its own bucket:

- Sending messages: ~5 per 5 seconds per channel
- Editing messages: ~5 per 5 seconds
- Global: 50 requests per second across all routes

discord.py handles rate limits automatically — it queues requests and retries after the rate limit window. You rarely need to handle 429s manually, but awareness helps when designing features that make many API calls.

## Description

Build a fully functional Discord bot using discord.py 2.4+. The practice covers five exercises progressing from basic Gateway connection through slash commands, modular Cog architecture, interactive UI components (buttons, modals), and event-driven role assignment. The bot connects to a real Discord test server — no emulators or mocks.

## Instructions

### Prerequisites: Discord Developer Portal Setup

Before touching code, set up the bot account:

1. **Create a Discord Application**: Go to [discord.com/developers/applications](https://discord.com/developers/applications) -> "New Application" -> name it (e.g., "Practice082Bot")
2. **Create a Bot user**: In the application settings, go to "Bot" -> "Add Bot" -> copy the **Token** (you'll need this in `.env`)
3. **Enable Privileged Intents**: On the Bot page, enable:
   - **Server Members Intent** (for `on_member_join`)
   - **Message Content Intent** (for reading message content)
4. **Generate OAuth2 Invite URL**: Go to "OAuth2" -> "URL Generator":
   - Scopes: `bot` + `applications.commands`
   - Bot Permissions: `Administrator` (for testing — in production you'd use minimal permissions)
   - Copy the generated URL
5. **Invite the bot**: Open the URL in a browser, select your test server, authorize
6. **Create a test server**: If you don't have one, create a Discord server for testing. Note its **Server ID** (enable Developer Mode in Discord settings -> right-click server -> "Copy Server ID")

### Project Setup

```bash
uv init --no-readme
uv add "discord.py>=2.4" python-dotenv
```

Then create `.env` from `.env.example`:
```bash
cp .env.example .env
# Edit .env with your actual token and guild ID
```

### Exercise 1 — Bot Skeleton & Gateway Events (~15 min)

**What you'll learn**: The event-driven push model. Discord pushes events via WebSocket; your bot registers async handlers. `Intents` act as a subscription mask — enabling/disabling entire categories of events.

Open `bot.py`. The scaffold provides the `DiscordBot` class (subclassing `commands.Bot`), logging setup, and the entry point. You need to implement:

1. **`configure_intents()`** — Create an `Intents` object with the right flags enabled. Think about which privileged intents this bot needs and why.
2. **`on_ready()`** — Log useful info when the bot connects (guild count, bot user name). This fires when the Gateway handshake completes.
3. **`on_member_join()`** — Send a welcome message when a new member joins. This requires the `members` privileged intent — without it, this event never fires.

### Exercise 2 — Slash Commands with Parameters & Guild Sync (~20 min)

**What you'll learn**: The modern `app_commands` API. Commands are registered with Discord's API (not just your bot), enabling auto-complete, type validation, and permission integration in the Discord client.

Open `bot.py`. Implement:

1. **`sync` command** — An admin-only slash command that calls `self.bot.tree.sync(guild=...)`. Without syncing, new commands don't appear in Discord. Guild sync is instant; global sync takes up to 1 hour.
2. **`ping` command** — Basic latency check. Teaches the `interaction.response.send_message()` pattern.
3. **`userinfo` command** — Takes a `discord.Member` parameter. Discord auto-completes member names in the UI and passes the resolved Member object to your handler.
4. **`serverinfo` command** — Uses `interaction.guild` to extract server metadata. Demonstrates how slash commands access guild context.

### Exercise 3 — Cogs: Modular Command Groups (~20 min)

**What you'll learn**: The Plugin/Extension pattern. Each Cog is a self-contained module loaded at runtime. This is how production bots organize hundreds of commands into manageable units.

1. **Move commands to Cogs**: Refactor the slash commands from Exercise 2 into `cogs/info.py` (ping, userinfo, serverinfo) and `cogs/moderation.py` (kick, ban stubs).
2. **Dynamic loading**: Implement `setup_hook()` in the Bot class to load extensions via `bot.load_extension()`.
3. **Module-level `setup()`**: Each cog file needs an `async def setup(bot)` entry point that discord.py calls when loading the extension.

### Exercise 4 — Views, Buttons & Modals (~25 min)

**What you'll learn**: Stateful async UI components. Views maintain state (vote counts, user tracking) and handle callbacks when users interact with buttons. Modals collect structured input via popup forms.

Open `cogs/interactions.py`. Implement:

1. **`PollView`** — A `discord.ui.View` with Yes/No buttons that track votes in-memory. Each button's callback updates the vote count and edits the original message. Handle duplicate votes (one vote per user).
2. **`FeedbackModal`** — A `discord.ui.Modal` with TextInput fields. The `on_submit` callback processes the form data and responds to the user.
3. **`/poll` command** — Creates a PollView and sends it as a message.
4. **`/feedback` command** — Opens the FeedbackModal.

### Exercise 5 — Rich Embeds & Reaction Roles (~10 min)

**What you'll learn**: Event correlation — connecting a reaction event to a role assignment action. Also covers rich embeds for visually structured messages.

In `cogs/interactions.py`, implement:

1. **`/roles_embed` command** — Sends a `discord.Embed` with emoji-to-role mappings. Uses color, fields, and footer.
2. **`on_raw_reaction_add` listener** — When a user reacts to the roles embed message, assign the corresponding role. Uses `RawReactionActionEvent` (works even if the message isn't cached).

## Motivation

Discord bot development exercises several production-relevant skills:

- **Event-driven architecture**: The Gateway/WebSocket model mirrors real-world event streaming (Kafka, Pub/Sub)
- **Async Python mastery**: discord.py is fully async — forces proper understanding of `async/await`, event loops, and concurrent state management
- **Plugin architecture**: Cogs demonstrate the Extension/Plugin pattern used in production systems
- **Interactive UI design**: Views and Modals teach stateful component lifecycle management
- **API integration**: OAuth2, rate limits, and permission models are universal patterns across SaaS APIs
- **Real-world deployment**: Unlike emulated practices, this bot connects to a real service — exposing you to actual API behavior, latency, and failure modes

## Commands

| Command | Description |
|---------|-------------|
| `uv init --no-readme` | Initialize Python project |
| `uv add "discord.py>=2.4" python-dotenv` | Install dependencies |
| `cp .env.example .env` | Create environment file from template |
| `uv run bot.py` | Start the bot (connects to Discord Gateway) |
| `uv run bot.py 2>&1 \| tee bot.log` | Start bot with logging to file and console |
| `python clean.py` | Remove generated files (__pycache__, .env) |

### Discord-side commands (run in Discord chat after bot is online)

| Command | Description |
|---------|-------------|
| `/sync` | Sync slash commands to the test guild (run once after adding new commands) |
| `/ping` | Check bot latency |
| `/userinfo [member]` | Display info about a server member |
| `/serverinfo` | Display server metadata |
| `/kick <member> [reason]` | Kick a member (requires permissions) |
| `/ban <member> [reason]` | Ban a member (requires permissions) |
| `/poll <question>` | Create an interactive poll with Yes/No buttons |
| `/feedback` | Open a feedback form modal |
| `/roles_embed` | Send the reaction-roles embed message |
