"""
Info Cog — Exercise 3

Groups informational slash commands (/ping, /userinfo, /serverinfo) into
a self-contained Cog. This file is loaded as an extension by the bot's
setup_hook() via bot.load_extension("cogs.info").

Architecture:
    InfoCog(commands.Cog)
    ├── /ping       → latency check
    ├── /userinfo   → member information embed
    └── /serverinfo → guild information embed

Each command is decorated with @app_commands.command() — when the Cog is
loaded, these commands are automatically added to the bot's CommandTree.
"""

from __future__ import annotations

import logging

import discord
from discord import app_commands
from discord.ext import commands

log = logging.getLogger("discord_bot.cogs.info")


class InfoCog(commands.Cog):
    """Informational commands — ping, user info, server info.

    This Cog demonstrates the Plugin/Extension pattern:
    - Commands are defined as methods of the Cog class
    - The Cog is loaded/unloaded at runtime
    - State (self.bot) is encapsulated within the Cog
    """

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(name="ping", description="Check bot latency")
    async def ping(self, interaction: discord.Interaction) -> None:
        """Respond with the bot's WebSocket latency.

        # TODO(human): Implement /ping (same logic as Exercise 2, now inside a Cog).
        #
        # WHAT: Reply with the bot's Gateway WebSocket latency in milliseconds.
        #
        # KEY DIFFERENCE FROM EXERCISE 2: Inside a Cog, you access the bot via
        # self.bot instead of interaction.client. Both work, but self.bot is the
        # idiomatic pattern in Cogs — it's set in __init__ and gives you the
        # typed Bot instance directly.
        #
        # HOW:
        #   1. Get latency: self.bot.latency (float, in seconds)
        #   2. Convert to ms: round(self.bot.latency * 1000)
        #   3. Respond: await interaction.response.send_message(f"Pong! {ms}ms")
        #
        # NOTE ON COGS: Notice the `self` parameter — this is a bound method of
        # the Cog class, not a free function. discord.py automatically strips
        # `self` when registering the command with Discord's API, so users still
        # see /ping with no extra parameters.
        #
        # Expected: "/ping" → "Pong! 42ms"
        """
        raise NotImplementedError("Exercise 3: InfoCog.ping")

    @app_commands.command(name="userinfo", description="Display info about a server member")
    @app_commands.describe(member="The member to inspect (defaults to yourself)")
    @app_commands.guild_only()
    async def userinfo(
        self,
        interaction: discord.Interaction,
        member: discord.Member | None = None,
    ) -> None:
        """Show information about a guild member.

        # TODO(human): Implement /userinfo (same logic as Exercise 2, now in a Cog).
        #
        # WHAT: Display member info — name, ID, join date, creation date, top role,
        # avatar — in a rich Embed.
        #
        # HOW (recap from Exercise 2):
        #   1. Default to invoker: target = member or interaction.user
        #      assert isinstance(target, discord.Member)
        #   2. Build discord.Embed with:
        #      - title=target.display_name
        #      - color=target.top_role.color
        #   3. Add fields: Username, ID, Joined Server, Account Created, Top Role
        #      Use discord.utils.format_dt(dt, "R") for relative timestamps
        #   4. Set thumbnail: embed.set_thumbnail(url=target.display_avatar.url)
        #   5. Send: await interaction.response.send_message(embed=embed)
        #
        # COG ADVANTAGE: If you later need to add caching or rate limiting to user
        # lookups, you can add state to the Cog (e.g., self.cache = {}) and share
        # it across all commands in this Cog. Each Cog encapsulates its own state.
        #
        # Expected: "/userinfo @someone" → embed with member details
        """
        raise NotImplementedError("Exercise 3: InfoCog.userinfo")

    @app_commands.command(name="serverinfo", description="Display server metadata")
    @app_commands.guild_only()
    async def serverinfo(self, interaction: discord.Interaction) -> None:
        """Show information about the current server.

        # TODO(human): Implement /serverinfo (same logic as Exercise 2, now in a Cog).
        #
        # WHAT: Display server name, owner, member count, channels, roles,
        # creation date, boost level — in a rich Embed.
        #
        # HOW (recap from Exercise 2):
        #   1. guild = interaction.guild (assert not None — @guild_only)
        #   2. Build discord.Embed:
        #      - title=guild.name
        #      - description=guild.description or "No description set"
        #      - color=discord.Color.blurple()
        #   3. Add fields (inline=True): Owner, Members, Channels, Roles, Created, Boost Level
        #   4. Thumbnail: guild.icon.url if guild.icon else skip
        #   5. Footer: f"Server ID: {guild.id}"
        #   6. Send: await interaction.response.send_message(embed=embed)
        #
        # Expected: "/serverinfo" → embed with server stats
        """
        raise NotImplementedError("Exercise 3: InfoCog.serverinfo")


# ===================================================================
# Extension entry point — discord.py calls this when loading the cog
# ===================================================================

async def setup(bot: commands.Bot) -> None:
    """Register the InfoCog with the bot.

    # TODO(human): Implement the extension setup function.
    #
    # WHAT: This is the entry point that discord.py calls when you run
    # bot.load_extension("cogs.info"). It must add the Cog to the bot.
    #
    # WHY: The setup() function is the "contract" between an extension file
    # and the bot. discord.py expects every extension to have an async function
    # named `setup` that takes the bot as its only parameter. This function
    # is responsible for registering any Cogs, commands, or listeners the
    # extension provides.
    #
    # HOW:
    #   1. Create the Cog instance: cog = InfoCog(bot)
    #   2. Add it to the bot: await bot.add_cog(cog)
    #   3. Log that the cog was loaded: log.info("InfoCog loaded")
    #
    # IMPORTANT: This MUST be async (async def setup). discord.py 2.0+ requires
    # async setup functions. Using a synchronous def setup() will raise an error.
    #
    # LIFECYCLE: When bot.load_extension("cogs.info") is called:
    #   1. Python imports the module (cogs/info.py)
    #   2. discord.py calls setup(bot)
    #   3. add_cog() registers all commands and listeners in the Cog
    #   4. Commands are now part of bot.tree and will be synced next time
    #
    # Expected: InfoCog is registered and its 3 commands are in the tree
    """
    log.info("Info cog skipped. Part of bot.py")
