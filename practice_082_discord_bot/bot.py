"""
Discord Bot — Entry Point

This module defines the DiscordBot class (subclassing commands.Bot), configures
intents, loads cog extensions, and starts the Gateway connection.

Architecture:
    DiscordBot (commands.Bot)
    ├── setup_hook()       → loads cog extensions before connecting
    ├── on_ready()         → fires when Gateway handshake completes
    ├── on_member_join()   → fires when a new member joins a guild
    └── CommandTree        → holds all slash commands (auto-created by commands.Bot)

Run with: uv run bot.py
"""

from __future__ import annotations

import logging
import os
import sys

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging — use the logging module, never print()
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("discord_bot")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

TOKEN: str | None = os.getenv("DISCORD_TOKEN")
TEST_GUILD_ID: str | None = os.getenv("TEST_GUILD_ID")

if not TOKEN:
    log.error("DISCORD_TOKEN not set in .env — cannot start bot.")
    sys.exit(1)

if not TEST_GUILD_ID:
    log.error("TEST_GUILD_ID not set in .env — cannot sync guild commands.")
    sys.exit(1)

TEST_GUILD = discord.Object(id=int(TEST_GUILD_ID))

# ---------------------------------------------------------------------------
# Cog extensions to load — each corresponds to a file in cogs/
# ---------------------------------------------------------------------------
EXTENSIONS: list[str] = [
    "cogs.info",
    "cogs.moderation",
    "cogs.interactions",
]


# ===================================================================
# Exercise 1 — Bot Skeleton & Gateway Events
# ===================================================================


def configure_intents() -> discord.Intents:
    """Create and return the Intents the bot needs.

    # TODO(human): Configure the bot's intent subscription mask.
    #
    # WHAT: Create a discord.Intents object with the right flags enabled.
    #
    # WHY: Intents control which events Discord sends to your bot over the
    # Gateway WebSocket. Without the right intents:
    #   - Missing `message_content` → message.content is always empty string
    #   - Missing `members` → on_member_join never fires, Member objects lack data
    #   - Missing `guilds` → no guild/channel events at all
    #
    # HOW:
    #   1. Start with discord.Intents.default() — this enables all NON-privileged
    #      intents (guilds, guild_messages, guild_reactions, etc.)
    #   2. Explicitly enable the privileged intents this bot needs:
    #      - message_content = True  (to read message text — Exercise 5 needs this)
    #      - members = True          (for on_member_join in this exercise)
    #   3. Return the configured Intents object
    #
    # PRIVILEGED INTENTS REMINDER: You must ALSO enable these in the Discord
    # Developer Portal → Bot → Privileged Gateway Intents. If you only enable
    # them in code but not in the portal, the bot will fail to connect with
    # a "disallowed intents" error.
    #
    # ANALOGY: Think of intents like Kafka consumer group topic subscriptions.
    # default() subscribes to all "public" topics. Privileged intents are like
    # topics that require admin approval to subscribe to.
    #
    # Expected: Returns an Intents object with default + message_content + members
    """

    intent = discord.Intents.default()
    intent.message_content = True
    intent.members = True

    return intent


class DiscordBot(commands.Bot):
    """Main bot class. Subclasses commands.Bot which provides:
    - CommandTree (self.tree) for slash commands
    - load_extension() for Cog loading
    - setup_hook() lifecycle method
    """

    def __init__(self) -> None:
        intents = configure_intents()
        super().__init__(
            command_prefix="!",  # Fallback prefix — we primarily use slash commands
            intents=intents,
            description="Practice 082 Discord Bot",
        )

    # ---------------------------------------------------------------
    # Exercise 3 — Cog Loading (implement after Exercises 1 & 2)
    # ---------------------------------------------------------------
    async def setup_hook(self) -> None:
        """Called before the bot connects to the Gateway. Load extensions here.

        # TODO(human): Load all cog extensions and sync commands to the test guild.
        #
        # WHAT: Iterate over the EXTENSIONS list and call self.load_extension()
        # for each one, then sync the command tree to the test guild.
        #
        # WHY: setup_hook() runs BEFORE the IDENTIFY payload is sent to Discord's
        # Gateway. This guarantees all commands are registered before the bot goes
        # online. If you load extensions in on_ready() instead, there's a race
        # condition — the bot might receive interactions before commands are registered.
        #
        # HOW:
        #   1. Loop over EXTENSIONS list
        #   2. For each extension, call: await self.load_extension(extension_name)
        #   3. Log each successful load
        #   4. Handle exceptions — if one cog fails, log the error but continue
        #      loading others (don't let one broken cog prevent the bot from starting)
        #   5. After all extensions are loaded, sync commands to the test guild:
        #      self.tree.copy_global_to(guild=TEST_GUILD)
        #      await self.tree.sync(guild=TEST_GUILD)
        #   6. Log how many commands were synced
        #
        # DESIGN DECISION: We use copy_global_to() + sync(guild=...) for instant
        # guild sync during development. In production, you'd use tree.sync()
        # (global) which takes up to 1 hour but works across all servers.
        #
        # ERROR HANDLING: Wrap each load_extension() in try/except. A common error
        # is ImportError when a cog file has syntax errors. Log the full traceback
        # so you can diagnose issues quickly.
        #
        # Expected: All 3 cogs load successfully, commands appear in Discord
        """

        for extension in EXTENSIONS:
            try:
                await self.load_extension(extension)
                log.info(f"Extension {extension} successfully loaded")
            except Exception as e:
                log.exception(f"Extension {extension} failed")
        self.tree.copy_global_to(guild=TEST_GUILD)
        synced = await self.tree.sync(guild=TEST_GUILD)
        log.info(f"{synced} commands synced")

    # ---------------------------------------------------------------
    # Exercise 1 — Gateway Events
    # ---------------------------------------------------------------
    async def on_ready(self) -> None:
        """Fires when the bot has successfully connected to the Gateway.

        # TODO(human): Log useful connection information.
        #
        # WHAT: Log the bot's username, discriminator, ID, and the number of
        # guilds (servers) it's connected to.
        #
        # WHY: on_ready confirms the WebSocket handshake completed and the bot
        # received its initial state (guild list, user info). This is your
        # "boot complete" signal. Note: on_ready can fire MULTIPLE TIMES if
        # the bot reconnects after a disconnect — don't put one-time setup here
        # (use setup_hook instead).
        #
        # HOW:
        #   1. Access self.user to get the bot's User object
        #      - self.user.name → bot username (e.g., "Practice082Bot")
        #      - self.user.id → bot's unique snowflake ID
        #   2. Access self.guilds to get the list of connected guilds
        #      - len(self.guilds) → number of servers
        #   3. Log all of this using the `log` logger (not print!)
        #   4. Optionally log each guild name for debugging
        #
        # IMPORTANT: self.user can technically be None before on_ready, but
        # inside on_ready it's guaranteed to be set. Use an assert or guard.
        #
        # Expected log output:
        #   "Bot connected as Practice082Bot (ID: 123456789). Guilds: 1"
        """

        log.info(f"Bot conneceted as {self.user.name} (ID:{self.user.id}). Guilds: {len(self.guilds)}")
        log.debug(f"Bot conneted to guilds: {self.guilds}")

    async def on_member_join(self, member: discord.Member) -> None:
        """Fires when a new member joins a guild the bot is in.

        # TODO(human): Send a welcome message to the guild's system channel.
        #
        # WHAT: When a member joins, send a greeting message mentioning them
        # in the guild's designated system channel.
        #
        # WHY: This demonstrates the event-driven model — you don't poll for
        # new members; Discord pushes the event to you. It also shows how
        # intents gate events: without `members` intent, this handler NEVER
        # fires, even though it's registered.
        #
        # HOW:
        #   1. Get the guild's system channel: member.guild.system_channel
        #      - This is the channel set in Server Settings → Overview → System Channel
        #      - It can be None if the server hasn't configured one
        #   2. Guard against None: if no system channel, log a warning and return
        #   3. Send a welcome message using system_channel.send()
        #      - Use member.mention to @-mention the new member
        #      - Example: f"Welcome to the server, {member.mention}!"
        #   4. Wrap in try/except for discord.HTTPException (bot might lack
        #      permission to send messages in that channel)
        #
        # TESTING: To test this, have a friend join your test server, or create
        # a second Discord account and join with it.
        #
        # INTENT REQUIREMENT: This event requires Intents.members = True AND
        # the "Server Members Intent" toggle enabled in the Developer Portal.
        # If either is missing, this handler silently never fires.
        #
        # Expected: New member gets an @-mention welcome in the system channel
        """

        system_channel = member.guild.system_channel

        if system_channel is None:
            log.warning(f"System channel is None. Server {member.guild.id} unconfigured")
            return

        try:
            await system_channel.send(f"Welcome to the server, {member.mention}!")
            log.debug(f"{member.name} was welcomed to {member.guild.id}")
        except Exception as e:
            log.exception(f"Exception welcoming {member.name} ({member.id}) to {member.guild.id}")

# ===================================================================
# Exercise 2 — Slash Commands (defined on the bot's tree)
#
# NOTE: After Exercise 3 (Cogs), these commands move to cogs/info.py.
# For Exercise 2, define them here directly on the bot instance.
# Once you complete Exercise 3, you can delete them from here since
# the Cog versions will take over.
# ===================================================================

# The bot instance — commands are defined below and registered on bot.tree
bot = DiscordBot()


@bot.tree.command(name="sync", description="Sync slash commands to this server (admin only)")
@app_commands.default_permissions(administrator=True)
@app_commands.guild_only()
async def sync_commands(interaction: discord.Interaction) -> None:
    """Manually sync the command tree to the current guild.

    # TODO(human): Implement the /sync command.
    #
    # WHAT: Sync the bot's command tree to the guild where this command is run.
    # This is an admin-only utility command used during development.
    #
    # WHY: When you add or modify slash commands, Discord doesn't automatically
    # know about the changes. You must call tree.sync() to push the current
    # command tree to Discord's API. Without syncing:
    #   - New commands don't appear in the Discord UI
    #   - Removed commands still show up (stale)
    #   - Changed parameters/descriptions don't update
    #
    # HOW:
    #   1. Get the current guild: interaction.guild
    #   2. Guard against None (shouldn't happen with @guild_only, but be safe)
    #   3. Call: synced = await interaction.client.tree.sync(guild=interaction.guild)
    #      - tree.sync() returns a list of synced AppCommand objects
    #   4. Respond with how many commands were synced:
    #      await interaction.response.send_message(f"Synced {len(synced)} commands.")
    #   5. Make the response ephemeral (only visible to the admin):
    #      Use ephemeral=True in send_message()
    #
    # GUILD SYNC vs GLOBAL SYNC:
    #   - tree.sync(guild=guild) → instant, only this server
    #   - tree.sync() → up to 1 hour, all servers
    #   During development, ALWAYS use guild sync for fast iteration.
    #
    # PERMISSION: The @default_permissions(administrator=True) decorator tells
    # Discord to only show this command to users with Administrator permission.
    # This is a UI hint — Discord hides the command from non-admins. For actual
    # enforcement, you'd add a check too, but for development this suffices.
    #
    # Expected: "/sync" responds with "Synced N commands to <guild_name>."
    """

    guild = interaction.guild

    if guild is None:
        log.error("Sync with guild=None")
        return

    synced = await interaction.client.tree.sync(guild=guild)
    await interaction.response.send_message(f"Synced {len(synced)} commands.", ephemeral=True)



@bot.tree.command(name="ping", description="Check bot latency")
async def ping(interaction: discord.Interaction) -> None:
    """Respond with the bot's WebSocket latency.

    # TODO(human): Implement the /ping command.
    #
    # WHAT: Reply with the bot's current Gateway WebSocket latency in milliseconds.
    #
    # WHY: A classic "hello world" for bots. It verifies the bot is responsive
    # and teaches the basic interaction.response.send_message() pattern. The
    # latency value comes from the WebSocket heartbeat — Discord sends periodic
    # heartbeat ACKs, and discord.py measures the round-trip time.
    #
    # HOW:
    #   1. Get latency: interaction.client.latency (float, in seconds)
    #   2. Convert to ms: round(interaction.client.latency * 1000)
    #   3. Respond: await interaction.response.send_message(f"Pong! {ms}ms")
    #
    # NOTE: interaction.client gives you the Bot instance. In a Cog, you'd
    # use self.bot instead, but here we're defining commands on the module level.
    #
    # Expected: "/ping" → "Pong! 42ms" (actual latency varies)
    """

    latency_ms = round(interaction.client.latency * 1000)

    await interaction.response.send_message(f"Pong! {latency_ms} ms")


@bot.tree.command(name="userinfo", description="Display info about a server member")
@app_commands.describe(member="The member to inspect (defaults to yourself)")
@app_commands.guild_only()
async def userinfo(
    interaction: discord.Interaction,
    member: discord.Member | None = None,
) -> None:
    """Show information about a guild member.

    # TODO(human): Implement the /userinfo command.
    #
    # WHAT: Display information about a server member — name, ID, join date,
    # account creation date, top role, and avatar.
    #
    # WHY: This teaches typed slash command parameters. When you type-hint
    # a parameter as discord.Member, Discord's UI shows a member picker with
    # auto-complete. Discord resolves the member and passes the full Member
    # object to your handler — no manual parsing needed.
    #
    # HOW:
    #   1. Default to the command invoker if no member specified:
    #      target = member or interaction.user
    #      assert isinstance(target, discord.Member)  # guild_only guarantees this
    #
    #   2. Build a discord.Embed with the member's info:
    #      embed = discord.Embed(
    #          title=target.display_name,
    #          color=target.top_role.color,  # Use their highest role's color
    #      )
    #
    #   3. Add fields to the embed:
    #      - "Username": target.name
    #      - "ID": target.id
    #      - "Joined Server": discord.utils.format_dt(target.joined_at, "R")
    #        (format_dt with "R" gives relative time like "2 months ago")
    #      - "Account Created": discord.utils.format_dt(target.created_at, "R")
    #      - "Top Role": target.top_role.mention
    #
    #   4. Set the thumbnail to the member's avatar:
    #      embed.set_thumbnail(url=target.display_avatar.url)
    #
    #   5. Send: await interaction.response.send_message(embed=embed)
    #
    # TYPE COERCION: discord.py + Discord API handle the Member resolution.
    # The user types a name → Discord auto-completes → resolves to Member →
    # passes Member object to your function. This is why slash commands are
    # superior to prefix commands where you'd manually parse "@username".
    #
    # Expected: "/userinfo @someone" → embed with avatar, join date, role info
    """

    target = member or interaction.user

    assert isinstance(target, discord.Member)

    embed = discord.Embed(
        title=target.display_name,
        color=target.top_role.color
    )

    embed.add_field(name="Username",value=target.name)
    embed.add_field(name="ID",value=target.id)
    embed.add_field(name="Joined Server",value=discord.utils.format_dt(target.joined_at,"R"))
    embed.add_field(name="Account Created",value=discord.utils.format_dt(target.created_at,"R"))
    embed.add_field(name="Top Role", value=target.top_role.mention)

    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="serverinfo", description="Display server metadata")
@app_commands.guild_only()
async def serverinfo(interaction: discord.Interaction) -> None:
    """Show information about the current server (guild).

    # TODO(human): Implement the /serverinfo command.
    #
    # WHAT: Display server name, ID, owner, member count, channel count,
    # creation date, and server icon.
    #
    # WHY: Demonstrates how slash commands access guild context via
    # interaction.guild. Also practices building rich Embeds — the primary
    # way bots present structured information in Discord.
    #
    # HOW:
    #   1. Get the guild: guild = interaction.guild
    #      assert guild is not None  # @guild_only guarantees this
    #
    #   2. Build an embed:
    #      embed = discord.Embed(
    #          title=guild.name,
    #          description=guild.description or "No description set",
    #          color=discord.Color.blurple(),  # Discord's signature color
    #      )
    #
    #   3. Add fields (use inline=True for side-by-side layout):
    #      - "Owner": guild.owner.mention if guild.owner else "Unknown"
    #      - "Members": guild.member_count
    #      - "Channels": len(guild.channels)
    #      - "Roles": len(guild.roles)
    #      - "Created": discord.utils.format_dt(guild.created_at, "R")
    #      - "Boost Level": guild.premium_tier
    #
    #   4. Set the thumbnail to the server icon:
    #      if guild.icon:
    #          embed.set_thumbnail(url=guild.icon.url)
    #
    #   5. Set footer with server ID:
    #      embed.set_footer(text=f"Server ID: {guild.id}")
    #
    #   6. Send: await interaction.response.send_message(embed=embed)
    #
    # EMBED ANATOMY:
    #   - title: Large bold text at top
    #   - description: Normal text below title
    #   - fields: Key-value pairs (inline=True puts them side by side)
    #   - thumbnail: Small image in top-right corner
    #   - footer: Small text at bottom
    #   - color: Colored stripe on left edge of embed
    #
    # Expected: "/serverinfo" → embed with server icon, stats, creation date
    """

    guild = interaction.guild
    assert guild is not None

    embed = discord.Embed(
        title=guild.name,
        description=guild.description or "No description set",
        color=discord.Color.blurple()
    )

    embed.add_field(name = "Owner", value = guild.owner.mention if guild.owner else "Unknown")
    embed.add_field(name = "Members", value = guild.member_count)
    embed.add_field(name = "Channels", value = len(guild.channels))
    embed.add_field(name = "Roles", value = len(guild.roles))
    embed.add_field(name = "Created", value = discord.utils.format_dt(guild.created_at,"R"))
    embed.add_field(name = "Boost Level", value = guild.premium_tier)

    if guild.icon:
        embed.set_thumbnail(url=guild.icon.url)

    embed.set_footer(text=f"Server ID: {guild.id}")

    await interaction.response.send_message(embed=embed)


# ===================================================================
# Entry Point
# ===================================================================

def main() -> None:
    """Start the bot. Connects to Discord's Gateway via WebSocket."""
    log.info("Starting bot... (press Ctrl+C to stop)")
    bot.run(TOKEN, log_handler=None)  # log_handler=None → we handle logging ourselves


if __name__ == "__main__":
    main()
