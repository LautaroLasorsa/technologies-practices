"""
Moderation Cog — Exercise 3

Groups moderation slash commands (/kick, /ban) into a permission-guarded Cog.
These are stub implementations — they demonstrate permission decorators and
the Cog pattern, not actual moderation logic (which would require more careful
audit logging, confirmation dialogs, etc. in production).

Architecture:
    ModerationCog(commands.Cog)
    ├── /kick  → kick a member (requires kick_members permission)
    └── /ban   → ban a member (requires ban_members permission)
"""

from __future__ import annotations

import logging

import discord
from discord import app_commands
from discord.ext import commands

log = logging.getLogger("discord_bot.cogs.moderation")


class ModerationCog(commands.Cog):
    """Moderation commands — kick, ban.

    These commands use @default_permissions to control visibility in the
    Discord UI. Only users with the specified permissions see the commands.
    The bot also needs the corresponding permissions to execute them.
    """

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(name="kick", description="Kick a member from the server")
    @app_commands.describe(
        member="The member to kick",
        reason="Reason for the kick (shown in audit log)",
    )
    @app_commands.guild_only()
    @app_commands.default_permissions(kick_members=True)
    async def kick(
        self,
        interaction: discord.Interaction,
        member: discord.Member,
        reason: str | None = None,
    ) -> None:
        """Kick a member from the guild.

        # TODO(human): Implement the /kick command.
        #
        # WHAT: Kick the specified member from the server, with an optional reason
        # that appears in the server's audit log.
        #
        # WHY: This teaches permission-guarded commands and the hierarchy of checks:
        #   1. @default_permissions(kick_members=True) — UI filter: Discord hides
        #      the command from users without kick_members permission
        #   2. Bot's own permissions — the bot account needs kick_members too
        #   3. Role hierarchy — you can't kick members with a higher role than the bot
        #
        # HOW:
        #   1. Validate the action is safe:
        #      - Can't kick yourself: member.id == interaction.user.id → error
        #      - Can't kick the server owner: member.id == interaction.guild.owner_id → error
        #      - Can't kick someone with a higher role than the bot:
        #        member.top_role >= interaction.guild.me.top_role → error
        #
        #   2. Attempt the kick:
        #      await member.kick(reason=reason)
        #
        #   3. Respond with confirmation:
        #      await interaction.response.send_message(
        #          f"Kicked {member.mention}. Reason: {reason or 'No reason provided'}",
        #          ephemeral=True,
        #      )
        #
        #   4. Handle errors:
        #      - discord.Forbidden → bot lacks permissions
        #      - discord.HTTPException → generic API error
        #      In both cases, respond with an ephemeral error message.
        #
        # ROLE HIERARCHY: Discord enforces a strict role hierarchy. A user/bot can
        # only moderate members whose highest role is BELOW their own highest role.
        # interaction.guild.me gives you the bot's Member object in that guild, so
        # interaction.guild.me.top_role is the bot's highest role.
        #
        # AUDIT LOG: The `reason` parameter is recorded in the guild's audit log,
        # visible to admins in Server Settings → Audit Log. This is important for
        # accountability in production bots.
        #
        # Expected: "/kick @user Spamming" → kicks user, confirms in ephemeral msg
        """
        raise NotImplementedError("Exercise 3: ModerationCog.kick")

    @app_commands.command(name="ban", description="Ban a member from the server")
    @app_commands.describe(
        member="The member to ban",
        reason="Reason for the ban (shown in audit log)",
        delete_days="Days of messages to delete (0-7)",
    )
    @app_commands.guild_only()
    @app_commands.default_permissions(ban_members=True)
    async def ban(
        self,
        interaction: discord.Interaction,
        member: discord.Member,
        reason: str | None = None,
        delete_days: app_commands.Range[int, 0, 7] = 0,
    ) -> None:
        """Ban a member from the guild.

        # TODO(human): Implement the /ban command.
        #
        # WHAT: Ban the specified member and optionally delete their recent messages.
        #
        # WHY: Extends the /kick pattern with an additional typed parameter:
        # app_commands.Range[int, 0, 7]. This tells Discord to validate the input
        # is between 0 and 7 — the UI shows a slider/number input with these bounds.
        # This is a powerful feature of slash commands: the validation happens on
        # Discord's side before your code even runs.
        #
        # HOW:
        #   1. Same safety checks as /kick:
        #      - Can't ban yourself
        #      - Can't ban the server owner
        #      - Can't ban someone with a higher role than the bot
        #
        #   2. Attempt the ban:
        #      await member.ban(
        #          reason=reason,
        #          delete_message_days=delete_days,
        #      )
        #
        #   3. Respond with confirmation (ephemeral):
        #      f"Banned {member.mention}. Reason: {reason or 'No reason'}. "
        #      f"Deleted {delete_days} day(s) of messages."
        #
        #   4. Handle discord.Forbidden and discord.HTTPException.
        #
        # PARAMETER TYPE: app_commands.Range[int, 0, 7] is discord.py's way of
        # defining a bounded integer parameter. Discord's UI enforces the range.
        # Other Range types include Range[float, 0.0, 1.0] for decimal ranges
        # and Range[str, 1, 100] for string length bounds.
        #
        # delete_message_days tells Discord to retroactively delete messages from
        # the banned member in the last N days. 0 = don't delete any.
        #
        # Expected: "/ban @user Harassment 7" → bans user, deletes 7 days of msgs
        """
        raise NotImplementedError("Exercise 3: ModerationCog.ban")


# ===================================================================
# Extension entry point
# ===================================================================

async def setup(bot: commands.Bot) -> None:
    """Register the ModerationCog with the bot.

    # TODO(human): Implement the extension setup function.
    #
    # WHAT: Same pattern as cogs/info.py — create the Cog and add it to the bot.
    #
    # HOW:
    #   1. await bot.add_cog(ModerationCog(bot))
    #   2. log.info("ModerationCog loaded")
    #
    # Expected: ModerationCog is registered, /kick and /ban appear in the tree
    """
    raise NotImplementedError("Exercise 3: setup for ModerationCog")
