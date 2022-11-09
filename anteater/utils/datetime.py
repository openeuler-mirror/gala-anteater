from datetime import datetime, timezone, timedelta

from anteater.utils.singleton import Singleton


class DateTimeManager(metaclass=Singleton):
    def __init__(self, *args, **kwargs):
        self.__freeze = False
        self.__freeze_utc_now = None
        super().__init__(*args, **kwargs)

    @property
    def utc_now(self) -> datetime:
        """Gets the current utc time"""
        if self.__freeze:
            return self.__freeze_utc_now
        else:
            return datetime.now(timezone.utc).astimezone()

    def update_and_freeze(self):
        self.__freeze = True
        self.__freeze_utc_now = datetime.now(timezone.utc).astimezone()

    def unfreeze(self):
        self.__freeze = False

    def last(self, seconds=0, minutes=0, hours=0):
        return (self.utc_now -
                timedelta(seconds=seconds, minutes=minutes, hours=hours)),\
               self.utc_now
