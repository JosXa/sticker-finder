"""Inline query handler function."""
from uuid import uuid4
from sqlalchemy.exc import IntegrityError
from telegram.ext import run_async
from telegram import InlineQueryResultCachedSticker

from stickerfinder.helper.session import hidden_session_wrapper
from stickerfinder.models import (
    InlineQuery,
    InlineQueryRequest,
)
from .context import Context
from .search import (
    search_stickers,
    search_sticker_sets,
)


@run_async
@hidden_session_wrapper()
def search(bot, update, session, user):
    """Handle inline queries for sticker search."""
    # We don't want banned users
    if user.banned:
        results = [InlineQueryResultCachedSticker(
            uuid4(),
            sticker_file_id='CAADAQADOQIAAjnUfAmQSUibakhEFgI')]
        update.inline_query.answer(results, cache_time=300, is_personal=True,
                                   switch_pm_text="Maybe don't be a dick :)?",
                                   switch_pm_parameter='inline')
        return

    offset_payload = update.inline_query.offset
    # If the offset is 'done' there are no more stickers for this query.
    if offset_payload == 'done':
        update.inline_query.answer([], cache_time=0)
        return

    context = Context(update.inline_query.query, offset_payload, user)

    # Create a new inline query or get the respective existing one, if we are working with an offset.
    inline_query = InlineQuery.get_or_create(session, context.inline_query_id, context.query, user)
    context.inline_query_id = inline_query.id

    if context.mode == Context.STICKER_SET_MODE:
        inline_query.mode = InlineQuery.SET_MODE

    # Save this specific InlineQueryRequest
    try:
        saved_offset = offset_payload.split(':', 1)[1] if context.offset != 0 else 0
        inline_query_request = InlineQueryRequest(inline_query, saved_offset)
        session.add(inline_query_request)
        session.commit()
    except IntegrityError:
        # This needs some explaining:
        # Sometimes (probably due to slow sticker loading) the telegram clients fire queries with the same offset.
        # To prevent this, we have an unique constraint on InlineQueryRequests.
        # If this constraint is violated, we assume that the scenario above just happened and just don't answer.
        # This prevents duplicate sticker suggestions due to slow internet connections.
        session.rollback()
        return

    if context.mode == Context.STICKER_SET_MODE:
        # Remove keyword tags to prevent wrong results
        search_sticker_sets(session, update, context, inline_query_request)
    else:
        search_stickers(session, update, context, inline_query_request)
