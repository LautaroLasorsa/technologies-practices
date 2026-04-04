"""
Interactions Cog — Exercises 4 & 5

Groups interactive UI components: polls with buttons, feedback modals,
rich embeds, and reaction-based role assignment.

Architecture:
    PollView(discord.ui.View)       → Yes/No vote buttons with in-memory tracking
    FeedbackModal(discord.ui.Modal) → Text input form for collecting feedback
    InteractionsCog(commands.Cog)
    ├── /poll <question>            → creates a PollView
    ├── /feedback                   → opens a FeedbackModal
    ├── /roles_embed                → sends reaction-role embed
    └── on_raw_reaction_add         → assigns role based on reaction
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

if TYPE_CHECKING:
    pass

log = logging.getLogger("discord_bot.cogs.interactions")


# ===================================================================
# Exercise 4a — Poll View with Buttons
# ===================================================================


class PollView(discord.ui.View):
    """A View with Yes/No buttons that tracks votes in-memory.

    Lifecycle:
        1. Created by the /poll command with a question
        2. Sent as part of a message (the View is "attached" to that message)
        3. Users click Yes/No buttons → callbacks fire
        4. After `timeout` seconds (default 180), buttons stop responding

    State:
        - votes: dict mapping "yes"/"no" to sets of user IDs
        - question: the poll question text

    # TODO(human): Implement the PollView class.
    #
    # WHAT: A discord.ui.View subclass with two buttons (Yes / No) that
    # track votes per user, prevent duplicate voting, and display live
    # vote counts by editing the original message.
    #
    # WHY: Views are discord.py's abstraction for stateful UI components.
    # Unlike slash commands (stateless request/response), Views maintain
    # state between interactions. This is conceptually similar to a React
    # component: the View has state (votes), renders UI (buttons), and
    # handles callbacks (button clicks) that update state and re-render.
    #
    # HOW — Step by step:
    #
    #   1. CONSTRUCTOR (__init__):
    #      def __init__(self, question: str, *, timeout: float = 180.0) -> None:
    #          super().__init__(timeout=timeout)
    #          self.question = question
    #          self.votes: dict[str, set[int]] = {"yes": set(), "no": set()}
    #
    #   2. YES BUTTON — use the @discord.ui.button() decorator:
    #      @discord.ui.button(label="Yes (0)", style=discord.ButtonStyle.green, custom_id="poll_yes")
    #      async def yes_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
    #          ...
    #
    #      Inside the callback:
    #      a. Get the user's ID: user_id = interaction.user.id
    #      b. Remove from opposite vote if they already voted:
    #         self.votes["no"].discard(user_id)
    #      c. Toggle: if already in "yes", remove (un-vote); otherwise add
    #      d. Update the button labels with current counts:
    #         button.label = f"Yes ({len(self.votes['yes'])})"
    #         Also update the No button — access it via self.children[1]
    #         (self.children is the list of components in order)
    #      e. Edit the message to reflect updated state:
    #         await interaction.response.edit_message(view=self)
    #
    #   3. NO BUTTON — same pattern as Yes, but for "no" votes:
    #      @discord.ui.button(label="No (0)", style=discord.ButtonStyle.red, custom_id="poll_no")
    #      async def no_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
    #          ...
    #
    #   4. TIMEOUT HANDLER — called when the view expires:
    #      async def on_timeout(self) -> None:
    #          # Disable all buttons so users see the poll is closed
    #          for child in self.children:
    #              if isinstance(child, discord.ui.Button):
    #                  child.disabled = True
    #          # self.message is set by discord.py if the view was sent with a message
    #          # But we need to store the message ourselves — see the /poll command
    #
    # DESIGN DECISIONS TO THINK ABOUT:
    #   - Should a user be able to un-vote (click the same button again to remove vote)?
    #     The scaffold above supports this via toggle logic.
    #   - Should a user be able to switch votes (click No after voting Yes)?
    #     The scaffold above supports this via discard from opposite set.
    #   - What happens on timeout? Buttons should be disabled and the message edited
    #     to show final results.
    #
    # STATE vs PERSISTENCE: The vote data lives in memory (self.votes). If the bot
    # restarts, all votes are lost. A production bot would persist to a database.
    # For this exercise, in-memory is fine.
    #
    # Expected behavior:
    #   1. /poll "Best language?" → message with "Best language?" + Yes(0)/No(0) buttons
    #   2. User clicks Yes → buttons update to Yes(1)/No(0)
    #   3. Same user clicks Yes again → un-votes: Yes(0)/No(0)
    #   4. After 180s → buttons become disabled
    """

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("Exercise 4: PollView.__init__")


# ===================================================================
# Exercise 4b — Feedback Modal
# ===================================================================


class FeedbackModal(discord.ui.Modal):
    """A modal dialog with text inputs for collecting user feedback.

    # TODO(human): Implement the FeedbackModal class.
    #
    # WHAT: A popup form with two text inputs — a short "subject" field and
    # a long "details" field. When submitted, the bot confirms receipt and
    # logs the feedback.
    #
    # WHY: Modals are Discord's way of collecting structured text input.
    # Unlike messages (free-form), modals have typed fields with validation
    # (required/optional, min/max length, placeholder text). Modals can only
    # be triggered from an interaction — you CANNOT send a modal unprompted.
    #
    # HOW — Step by step:
    #
    #   1. CLASS DEFINITION:
    #      class FeedbackModal(discord.ui.Modal, title="Submit Feedback"):
    #          ...
    #      The `title=` in the class declaration sets the modal's title bar.
    #
    #   2. TEXT INPUTS — define as class-level attributes:
    #
    #      subject = discord.ui.TextInput(
    #          label="Subject",
    #          placeholder="Brief summary of your feedback...",
    #          style=discord.TextStyle.short,   # Single line
    #          required=True,
    #          max_length=100,
    #      )
    #
    #      details = discord.ui.TextInput(
    #          label="Details",
    #          placeholder="Describe your feedback in detail...",
    #          style=discord.TextStyle.long,    # Multi-line text area
    #          required=False,
    #          max_length=1000,
    #      )
    #
    #   3. ON_SUBMIT — called when the user clicks "Submit":
    #      async def on_submit(self, interaction: discord.Interaction) -> None:
    #          a. Access values: self.subject.value, self.details.value
    #          b. Log the feedback: log.info(f"Feedback from {interaction.user}: ...")
    #          c. Respond (ephemeral): "Thanks for your feedback!"
    #          d. Optionally build an embed summarizing the feedback
    #
    #   4. ON_ERROR — handle exceptions during submission:
    #      async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
    #          log.exception("Error in FeedbackModal")
    #          await interaction.response.send_message(
    #              "Something went wrong. Please try again.", ephemeral=True
    #          )
    #
    # TextStyle ENUM:
    #   - TextStyle.short → single-line input (like HTML <input type="text">)
    #   - TextStyle.long  → multi-line textarea (like HTML <textarea>)
    #
    # MODAL LIMITS: Up to 5 TextInput components per modal. No buttons, no
    # select menus — only text inputs. This is a Discord API limitation.
    #
    # Expected:
    #   1. "/feedback" → modal popup with Subject + Details fields
    #   2. User fills in fields, clicks Submit
    #   3. Bot responds "Thanks for your feedback!" (ephemeral)
    #   4. Feedback is logged to console
    """

    pass  # Replace with your implementation


# ===================================================================
# Exercise 5 — Reaction Roles Configuration
# ===================================================================

# Mapping of emoji (Unicode) to role name.
# The /roles_embed command displays this mapping, and on_raw_reaction_add
# uses it to assign roles when users react.
#
# NOTE: These role names must match EXISTING roles in your test server.
# Create these roles manually in Server Settings → Roles before testing.
EMOJI_TO_ROLE: dict[str, str] = {
    "\U0001f534": "Red Team",     # Red circle emoji → "Red Team" role
    "\U0001f535": "Blue Team",    # Blue circle emoji → "Blue Team" role
    "\U0001f7e2": "Green Team",   # Green circle emoji → "Green Team" role
}


# ===================================================================
# Interactions Cog
# ===================================================================


class InteractionsCog(commands.Cog):
    """Interactive UI components — polls, feedback, reaction roles."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        # Store the ID of the reaction-role embed message so the listener
        # knows which message to watch. Set by /roles_embed command.
        self.role_message_id: int | None = self._load_role_message_id()

    @staticmethod
    def _load_role_message_id() -> int | None:
        """Load persisted role message ID from environment, if set."""
        raw = os.getenv("ROLE_MESSAGE_ID", "")
        if raw.strip():
            try:
                return int(raw)
            except ValueError:
                pass
        return None

    # ---------------------------------------------------------------
    # Exercise 4 — /poll command
    # ---------------------------------------------------------------

    @app_commands.command(name="poll", description="Create a poll with Yes/No buttons")
    @app_commands.describe(question="The question to ask")
    @app_commands.guild_only()
    async def poll(self, interaction: discord.Interaction, question: str) -> None:
        """Create a poll with interactive Yes/No buttons.

        # TODO(human): Implement the /poll command.
        #
        # WHAT: Create a PollView for the given question and send it as a message
        # with the view attached.
        #
        # WHY: This teaches how to attach a View to a message. The View object
        # lives in your bot's memory; the message in Discord displays the buttons.
        # When a user clicks a button, Discord sends an interaction to your bot,
        # which routes it to the correct View callback.
        #
        # HOW:
        #   1. Create the view: view = PollView(question=question)
        #
        #   2. Build an embed for the poll:
        #      embed = discord.Embed(
        #          title="Poll",
        #          description=question,
        #          color=discord.Color.gold(),
        #      )
        #      embed.set_footer(text=f"Started by {interaction.user.display_name}")
        #
        #   3. Send the message with the view:
        #      await interaction.response.send_message(embed=embed, view=view)
        #
        #   4. Store reference to the message for timeout handling:
        #      view.message = await interaction.original_response()
        #      This lets the view edit the message when it times out.
        #
        # LIFECYCLE: The View stays alive for `timeout` seconds (default 180).
        # During that time, button clicks are routed to the View's callbacks.
        # After timeout, on_timeout() fires and the View stops processing.
        # If the bot restarts, all active Views are lost (they're in-memory).
        #
        # Expected: "/poll Best language?" → embed + Yes(0)/No(0) buttons
        """
        raise NotImplementedError("Exercise 4: InteractionsCog.poll")

    # ---------------------------------------------------------------
    # Exercise 4 — /feedback command
    # ---------------------------------------------------------------

    @app_commands.command(name="feedback", description="Submit feedback via a form")
    async def feedback(self, interaction: discord.Interaction) -> None:
        """Open a feedback modal dialog.

        # TODO(human): Implement the /feedback command.
        #
        # WHAT: When invoked, send a FeedbackModal to the user as a popup dialog.
        #
        # WHY: This teaches the interaction.response.send_modal() pattern. Modals
        # are unique because they're the only response type that opens a NEW UI
        # element (a popup form). Other responses (send_message, edit_message)
        # modify the chat — modals overlay on top.
        #
        # HOW:
        #   1. Create the modal: modal = FeedbackModal()
        #   2. Send it: await interaction.response.send_modal(modal)
        #
        # That's it! The modal handles its own on_submit callback. The command
        # just triggers the popup.
        #
        # IMPORTANT: You can ONLY send a modal as the FIRST response to an
        # interaction. If you call send_message() first, you can't then send
        # a modal — you'll get InteractionResponded. Modals must be the
        # initial response.
        #
        # Expected: "/feedback" → popup form appears with Subject + Details fields
        """
        raise NotImplementedError("Exercise 4: InteractionsCog.feedback")

    # ---------------------------------------------------------------
    # Exercise 5 — /roles_embed command
    # ---------------------------------------------------------------

    @app_commands.command(name="roles_embed", description="Send the reaction-roles embed")
    @app_commands.default_permissions(administrator=True)
    @app_commands.guild_only()
    async def roles_embed(self, interaction: discord.Interaction) -> None:
        """Send a rich embed with emoji-to-role mappings, then add reactions.

        # TODO(human): Implement the /roles_embed command.
        #
        # WHAT: Send a visually rich Embed that lists emoji-to-role mappings,
        # then programmatically add those emoji as reactions to the message.
        # When users later react, on_raw_reaction_add assigns the role.
        #
        # WHY: This teaches two concepts:
        #   1. Rich Embeds — structured, visually appealing messages with colors,
        #      fields, footers, and images. Embeds are the primary way bots
        #      present formatted information.
        #   2. Seeding reactions — the bot adds initial reactions so users know
        #      which emoji to click. This is a UX pattern, not a technical
        #      requirement (users could react with any emoji).
        #
        # HOW:
        #   1. Build the embed:
        #      embed = discord.Embed(
        #          title="Role Selection",
        #          description="React with an emoji to get the corresponding role!",
        #          color=discord.Color.blue(),
        #      )
        #
        #   2. Add a field for each emoji-role mapping:
        #      for emoji, role_name in EMOJI_TO_ROLE.items():
        #          embed.add_field(name=emoji, value=role_name, inline=True)
        #
        #   3. Set footer with instructions:
        #      embed.set_footer(text="Remove your reaction to lose the role")
        #
        #   4. Send the embed (NOT ephemeral — everyone needs to see it):
        #      await interaction.response.send_message(embed=embed)
        #      message = await interaction.original_response()
        #
        #   5. Store the message ID for the reaction listener:
        #      self.role_message_id = message.id
        #      log.info(f"Role embed message ID: {message.id} — save to ROLE_MESSAGE_ID in .env")
        #
        #   6. Add seed reactions to the message:
        #      for emoji in EMOJI_TO_ROLE:
        #          await message.add_reaction(emoji)
        #
        # PERSISTENCE: The role_message_id is stored in memory. If the bot
        # restarts, it's lost UNLESS the user saves it to .env (ROLE_MESSAGE_ID).
        # The constructor checks for this env var on startup.
        #
        # Expected: "/roles_embed" → embed with emoji fields + bot adds reactions
        """
        raise NotImplementedError("Exercise 5: InteractionsCog.roles_embed")

    # ---------------------------------------------------------------
    # Exercise 5 — Reaction Role Listener
    # ---------------------------------------------------------------

    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent) -> None:
        """Assign a role when a user reacts to the roles embed.

        # TODO(human): Implement the reaction-role listener.
        #
        # WHAT: When a user adds a reaction to the roles embed message, look up
        # the corresponding role and assign it to the user.
        #
        # WHY: This teaches EVENT CORRELATION — connecting two separate events
        # (reaction add → role assignment) through a shared identifier (message ID).
        # It also demonstrates why we use on_RAW_reaction_add instead of
        # on_reaction_add:
        #   - on_reaction_add only fires if the message is in the bot's cache
        #   - on_raw_reaction_add fires for ANY reaction, even on uncached messages
        #   - For reaction roles, the embed might be hours/days old and not cached
        #
        # HOW:
        #   1. FILTER — only process reactions on the roles embed:
        #      if self.role_message_id is None:
        #          return
        #      if payload.message_id != self.role_message_id:
        #          return
        #
        #   2. IGNORE BOT REACTIONS — don't assign roles to the bot itself:
        #      if payload.user_id == self.bot.user.id:
        #          return
        #
        #   3. RESOLVE THE EMOJI — payload.emoji is a PartialEmoji. For Unicode
        #      emoji, compare with str(payload.emoji):
        #      emoji_str = str(payload.emoji)
        #      role_name = EMOJI_TO_ROLE.get(emoji_str)
        #      if role_name is None:
        #          return  # Unknown emoji, ignore
        #
        #   4. GET THE GUILD AND MEMBER:
        #      if payload.guild_id is None:
        #          return  # DM reaction, ignore
        #      guild = self.bot.get_guild(payload.guild_id)
        #      if guild is None:
        #          return
        #      # payload.member is available in on_raw_reaction_add (but NOT remove)
        #      member = payload.member
        #      if member is None:
        #          return
        #
        #   5. FIND THE ROLE by name:
        #      role = discord.utils.get(guild.roles, name=role_name)
        #      if role is None:
        #          log.warning(f"Role '{role_name}' not found in {guild.name}")
        #          return
        #
        #   6. ASSIGN THE ROLE:
        #      try:
        #          await member.add_roles(role, reason="Reaction role")
        #          log.info(f"Assigned '{role_name}' to {member}")
        #      except discord.Forbidden:
        #          log.error(f"Missing permissions to assign '{role_name}'")
        #      except discord.HTTPException as e:
        #          log.error(f"Failed to assign role: {e}")
        #
        # RAW vs CACHED EVENTS: The "raw" prefix means the event provides
        # minimal data (IDs, not full objects) but fires regardless of cache
        # state. This is essential for reaction roles because the roles embed
        # message might have been sent hours ago and evicted from cache.
        #
        # payload.member CAVEAT: The member field is populated in
        # on_raw_reaction_ADD (Discord includes it). In on_raw_reaction_REMOVE,
        # member is None — you'd need guild.get_member(payload.user_id) or
        # guild.fetch_member(payload.user_id) instead.
        #
        # Expected:
        #   1. User reacts with red circle on roles embed
        #   2. Bot assigns "Red Team" role to that user
        #   3. Log: "Assigned 'Red Team' to User#1234"
        """
        raise NotImplementedError("Exercise 5: InteractionsCog.on_raw_reaction_add")


# ===================================================================
# Extension entry point
# ===================================================================

async def setup(bot: commands.Bot) -> None:
    """Register the InteractionsCog with the bot.

    # TODO(human): Implement the extension setup function.
    #
    # WHAT: Same pattern as the other cogs — create and add the Cog.
    #
    # HOW:
    #   1. await bot.add_cog(InteractionsCog(bot))
    #   2. log.info("InteractionsCog loaded")
    #
    # Expected: InteractionsCog is registered with /poll, /feedback, /roles_embed
    """
    raise NotImplementedError("Exercise 5: setup for InteractionsCog")
