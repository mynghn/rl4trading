import datetime

from gym.spaces import Space


class Date(Space):
    def __init__(self, start: datetime.date, end: datetime.date):
        self.shape = (1,)
        self.dtype = datetime.date
        self._np_random = None

        self.start = start
        self.end = end

    def sample(self) -> datetime.date:
        return self.start + datetime.timedelta(
            days=self.np_random.randint((self.end - self.start).days + 1)
        )

    def contains(self, x: datetime.date) -> bool:
        return self.start <= x <= self.end
