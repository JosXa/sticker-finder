#!/bin/env python3
"""Helper script for testing stuff."""

import os
import sys


parent_dir = os.path.abspath(os.getcwd())
sys.path.append(parent_dir)

from stickerfinder.models import StickerSet
from stickerfinder.db import get_session
from stickerfinder.stickerfinder import updater

session = get_session()

sticker_set = session.query(StickerSet).get('cheloidesmemestash2')

sticker_set.refresh_stickers(session, updater.bot, refresh_ocr=True)
