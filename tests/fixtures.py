"""Database test fixtures."""
import pytest
from tests.factories import user_factory, sticker_set_factory, sticker_factory

from stickerfinder.helper.tag import tag_sticker
from stickerfinder.models import Sticker


@pytest.fixture(scope='function')
def user(session):
    """Create a user."""
    return user_factory(session, 2, 'TestUser')


@pytest.fixture(scope='function')
def admin(session):
    """Create a user."""
    return user_factory(session, 1, 'admin', True)


@pytest.fixture(scope='function')
def sticker_set(session, admin):
    """Create a user."""
    stickers = []
    for file_id in range(0, 10):
        sticker = Sticker(str(file_id))
        stickers.append(sticker)

    return sticker_set_factory(session, 'test_set', stickers)


@pytest.fixture(scope='function')
def tags(session, sticker_set, user):
    """Create tags for all stickers."""
    for sticker in sticker_set.stickers:
        # Create a new tag for each sticker
        tag_sticker(session, f'tag_{sticker.file_id}', sticker, user)


@pytest.fixture(scope='function')
def strict_inline_search(session):
    """Create several sticker sets and stickers with tags for strict sticker search testing."""
    # Create a set with a 40 stickers, each having one tag `testtag`
    sticker_set_1 = sticker_set_factory(session, 'z_mega_awesome')
    for i in range(0, 40):
        # This is a little workaround to prevent fucky number sorting stuff
        if i < 10:
            i = f'0{i}'
        sticker = sticker_factory(session, f'sticker_{i}', ['testtag', 'unique_other'])
        sticker_set_1.stickers.append(sticker)

    # Create a second set with 20 stickers, each having one tag `testtag` as well
    sticker_set_2 = sticker_set_factory(session, 'a_dumb_shit')
    for i in range(40, 60):
        sticker = sticker_factory(session, f'sticker_{i}', ['testtag', 'roflcopter'])
        sticker_set_2.stickers.append(sticker)
    session.commit()

#    # Debugg stuff
#    print(sticker_set_1)
#    for sticker in sticker_set_1.stickers:
#        print(sticker)
#    print(sticker_set_2)
#    for sticker in sticker_set_2.stickers:
#        print(sticker)

    return [sticker_set_1, sticker_set_2]
