"""The sqlite model for a sticker set."""
import io
import re
import logging
from PIL import Image
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import relationship
from telegram.error import BadRequest, TimedOut
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    func,
    Index,
    CheckConstraint,
)

from stickerfinder.db import base
from stickerfinder.sentry import sentry
from stickerfinder.models import chat_sticker_set, Sticker, Task
from stickerfinder.helper.telegram import call_tg_func
from stickerfinder.helper.image import old_get_text_from_image, get_text_from_image


class StickerSet(base):
    """The sqlite model for a sticker set."""

    __tablename__ = 'sticker_set'
    __table_args__ = (
        Index('sticker_set_name_gin_idx', 'name',
              postgresql_using='gin', postgresql_ops={'name': 'gin_trgm_ops'}),
        Index('sticker_title_name_gin_idx', 'title',
              postgresql_using='gin', postgresql_ops={'title': 'gin_trgm_ops'}),
        CheckConstraint("NOT (reviewed AND NOT complete)"),
    )

    name = Column(String, primary_key=True)
    title = Column(String)
    is_default_language = Column(Boolean, default=True, nullable=False)
    deleted = Column(Boolean, default=False, nullable=False)

    # Flags
    banned = Column(Boolean, default=False, nullable=False)
    nsfw = Column(Boolean, default=False, nullable=False)
    furry = Column(Boolean, default=False, nullable=False)

    # Metadata
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    complete = Column(Boolean, default=False, nullable=False)
    completely_tagged = Column(Boolean, default=False, nullable=False)
    reviewed = Column(Boolean, default=False, nullable=False)

    stickers = relationship("Sticker", order_by="desc(Sticker.file_id)")
    reports = relationship("Report", order_by="desc(Report.created_at)")
    tasks = relationship("Task")
    chats = relationship(
        "Chat",
        secondary=chat_sticker_set,
        back_populates="sticker_sets")

    def __init__(self, name, stickers):
        """Create a new StickerSet instance."""
        self.name = name
        self.stickers = []

    def refresh_stickers(self, session, bot, refresh_ocr=False, chat=None):
        """Refresh stickers and set data from telegram."""
        # Get sticker set from telegram and create new a Sticker for each sticker
        stickers = []
        logger = logging.getLogger()
        try:
            tg_sticker_set = call_tg_func(bot, 'get_sticker_set', args=[self.name])
        except BadRequest as e:
            if e.message == 'Stickerset_invalid': # noqa
                self.deleted = True
                return

            raise e

        for tg_sticker in tg_sticker_set.stickers:
            # Ignore already existing stickers if we don't need to rescan images
            sticker = session.query(Sticker).get(tg_sticker.file_id)
            text = None
            if sticker is None or refresh_ocr:
                try:
                    # Get Image and preprocess it
                    tg_file = call_tg_func(tg_sticker, 'get_file')
                    image_bytes = call_tg_func(tg_file, 'download_as_bytearray')

                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    with io.BytesIO() as output:
                        image.save(output, format="PNG")
                        contents = output.getvalue()

                    config = {
                        'min_confidence': 0.5,
                        'padding': 0.0,
                        'height': 320,
                        'width': 320,
                    }

                    text = old_get_text_from_image(image)
                    new_text = get_text_from_image(contents, config)
                    print(f'\nNext one: {sticker.file_id}')
                    print(f'Old: {text}')

                    # Only allow chars and remove multiple spaces to single spaces
                    text = re.sub('[^a-zA-Z ]+', '', text)
                    text = re.sub(' +', ' ', text)
                    text = text.strip()

                    print(f'Old cleaned: {text}')
                    print(f'New: {new_text}')
                    if text == '':
                        text = None

                except TimedOut:
                    logger.info(f'Finally failed on file {tg_sticker.file_id}')
                    pass
                except BadRequest:
                    logger.info(f'Failed to get image of f{tg_sticker.file_id}')
                    pass
                except BaseException as e:
                    raise e
                    sentry.captureException()
                    pass

            # Create new Sticker.
            if sticker is None:
                sticker = Sticker(tg_sticker.file_id)
                stickers.append(sticker)

            # Only set text, if we got some text from the ocr recognition
            if text is not None:
                sticker.text = text

            sticker.add_emojis(session, tg_sticker.emoji)
            session.commit()

        self.name = tg_sticker_set.name.lower()

        self.title = tg_sticker_set.title.lower()
        self.stickers = stickers
        self.complete = True
        session.commit()

    @staticmethod
    def get_or_create(session, name, chat, user):
        """Get or create a new sticker set."""
        name = name.lower()
        sticker_set = session.query(StickerSet).get(name)
        if not sticker_set:
            # Create a task for adding a sticker.
            # This task will be processed by a job, since adding a sticker can take quite a while
            sticker_set = StickerSet(name, None)
            sticker_set.is_default_language = user.is_default_language
            task = Task(Task.SCAN_SET, sticker_set=sticker_set, chat=chat, user=user)
            session.add(sticker_set)
            session.add(task)
            # Error handling: Retry in case somebody sent to stickers at the same time
            try:
                session.commit()
            except IntegrityError as e:
                session.rollback()
                sticker_set = session.query(StickerSet).get(name)
                if sticker_set is None:
                    raise e

        return sticker_set
