from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
from random import random
from .houses import House
from .house_market import HousingMarket

class Segment(Enum):
    FANCY = auto()  # House is new construction and house score is the highest
    OPTIMIZER = auto()  # Price per square foot is less than monthly salary
    AVERAGE = auto()  # House price is below the average housing market price

@dataclass
class Consumer:
    """Class representing a consumer (potential buyer) in the housing market."""
    id: int
    annual_income: float
    children_number: int
    segment: Segment
    house: Optional[House] = None
    savings: float = 0.0
    saving_rate: float = 0.3
    interest_rate: float = 0.05

    def compute_savings(self, years: int) -> None:
        """
        Calculate accumulated savings over a specified number of years.

        Args:
            years (int): The number of years for which savings will accumulate.
        """
        for _ in range(years):
            self.savings += self.saving_rate * self.annual_income
            self.savings *= (1 + self.interest_rate)

    def buy_a_house(self, housing_market: HousingMarket) -> None:
        """
        Attempt to purchase a suitable house based on the consumer's segment and preferences.

        Args:
            housing_market (HousingMarket): The housing market object containing available houses.
        """
        if self.house is not None:
            return  # Consumer already owns a house

        # Get houses that match consumer's preferences
        if self.segment == Segment.FANCY:
            matching_houses = [
                house for house in housing_market.houses
                if house.is_new_construction() and house.quality_score and house.quality_score.value >= 4
            ]
        elif self.segment == Segment.OPTIMIZER:
            matching_houses = [
                house for house in housing_market.houses
                if house.calculate_price_per_square_foot() < (self.annual_income / 12)
            ]
        else:  # Segment.AVERAGE
            matching_houses = [
                house for house in housing_market.houses
                if house.price < housing_market.calculate_average_price()
            ]

        # Sort houses by price (cheapest first)
        matching_houses.sort(key=lambda h: h.price)

        for house in matching_houses:
            down_payment = self.savings * 0.2
            if down_payment >= house.price * 0.2:  # Check if consumer can afford the down payment
                self.house = house
                house.sell_house()
                break


