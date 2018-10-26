"""The sqlite model for a change."""
from sqlalchemy.orm import relationship
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    func,
    Integer,
    String,
    ForeignKey,
)

from stickerfinder.db import base


class Change(base):
    """The model for a change."""

    __tablename__ = 'change'

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    reverted = Column(Boolean, server_default='false', default=False, nullable=False)
    old_tags = Column(String)
    new_tags = Column(String)

    old_text = Column(String)
    new_text = Column(String)

    user_id = Column(BigInteger, ForeignKey('user.id'), index=True)
    sticker_file_id = Column(String, ForeignKey('sticker.file_id', ondelete='cascade'), index=True)

    user = relationship("User")
    sticker = relationship("Sticker")

    def __init__(self, user, sticker, old_tags=None, old_text=None):
        """Create a new change."""
        self.user = user
        self.sticker = sticker

        if old_tags:
            self.old_tags = old_tags
            self.new_tags = sticker.tags_as_text()

        if old_text:
            self.old_text = old_text
            self.new_text = sticker.text
