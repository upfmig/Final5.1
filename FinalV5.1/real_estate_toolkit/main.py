# test jose

from pathlib import Path
import numpy as np
import polars as pl
from typing import List, Dict, Any


from src.real_estate_toolkit.data.loader import DataLoader
from src.real_estate_toolkit.data.cleaner import Cleaner
from src.real_estate_toolkit.data.descriptor import Descriptor, DescriptorNumpy
from src.real_estate_toolkit.agent_based_model.houses import House, QualityScore
from src.real_estate_toolkit.agent_based_model.house_market import HousingMarket
from src.real_estate_toolkit.agent_based_model.consumers import Consumer, Segment
from src.real_estate_toolkit.agent_based_model.simulation import (
    Simulation, 
    CleaningMarketMechanism, 
    AnnualIncomeStatistics,
    ChildrenRange
)

def test_data_loading_and_cleaning():
    """Test data loading and cleaning functionality"""
    # Test data loading
    data_path = Path("files/train.csv")
    loader = DataLoader(data_path)
    
    # Test column validation
    required_columns = ["Id", "SalePrice", "LotArea", "YearBuilt", "BedroomAbvGr"]
    assert loader.validate_columns(required_columns), "Required columns missing from dataset"
    
    # Load and test data format
    data = loader.load_data_from_csv()
    assert isinstance(data, list), "Data should be returned as a list"
    assert all(isinstance(row, dict) for row in data), "Each row should be a dictionary"
    
    # Test data cleaning
    cleaner = Cleaner(data)
    cleaner.rename_with_best_practices()
    cleaned_data = cleaner.na_to_none()
    
    # Verify cleaning results
    assert all(key.islower() and "_" in key for key in cleaned_data[0].keys()), "Column names should be in snake_case"
    assert all(val is None or isinstance(val, (str, int, float)) for row in cleaned_data for val in row.values()), \
        "Values should be None or basic types"
    
    return cleaned_data


def test_descriptive_statistics(cleaned_data: List[Dict[str, Any]]):
    """Test descriptive statistics functionality"""
    descriptor = Descriptor(cleaned_data)
    descriptor_numpy = DescriptorNumpy(cleaned_data)
    
    # Test none ratio calculation
    none_ratios = descriptor.none_ratio()
    none_ratios_numpy = descriptor_numpy.none_ratio()
    assert isinstance(none_ratios, dict), "None ratios should be returned as dictionary"
    assert set(none_ratios.keys()) == set(none_ratios_numpy.keys()), "Both implementations should handle same columns"
    
    # Test numeric calculations
    numeric_columns = ["sale_price", "lot_area"]  # Assuming these are the cleaned names
    averages = descriptor.average(numeric_columns)
    medians = descriptor.median(numeric_columns)
    percentiles = descriptor.percentile(numeric_columns, 75)
    
    # Test numpy implementation
    averages_numpy = descriptor_numpy.average(numeric_columns)
    medians_numpy = descriptor_numpy.median(numeric_columns)
    percentiles_numpy = descriptor_numpy.percentile(numeric_columns, 75)
    
    # Compare results
    for col in numeric_columns:
        assert abs(averages[col] - averages_numpy[col]) < 1e-6, f"Average calculations differ for {col}"
        assert abs(medians[col] - medians_numpy[col]) < 1e-6, f"Median calculations differ for {col}"
        assert abs(percentiles[col] - percentiles_numpy[col]) < 1e-6, f"Percentile calculations differ for {col}"
    
    # Test type and mode
    type_modes = descriptor.type_and_mode()
    type_modes_numpy = descriptor_numpy.type_and_mode()
    assert set(type_modes.keys()) == set(type_modes_numpy.keys()), "Both implementations should handle same columns"
    
    return numeric_columns

def test_house_functionality():
    """Test House class implementation"""
    house = House(
        id=1,
        price=200000.0,
        area=2000.0,
        bedrooms=3,
        year_built=2010,
        quality_score=QualityScore.GOOD,
        available=True
    )
    
    # Test basic calculations
    price_per_sqft = house.calculate_price_per_square_foot()
    assert isinstance(price_per_sqft, float), "Price per square foot should be float"
    assert price_per_sqft == 100.0, "Incorrect price per square foot calculation"
    
    # Test new construction logic
    assert house.is_new_construction(2024) is False, "House should not be considered new construction"
    assert house.is_new_construction(2012) is True, "House should be considered new construction"
    
    # Test quality score generation
    house.quality_score = None
    house.get_quality_score()
    assert house.quality_score is not None, "Quality score should be generated"
    
    # Test house sale
    house.sell_house()
    assert house.available is False, "House should be marked as unavailable after sale"
    
    return house

def test_market_functionality(house: House):
    """Test HousingMarket class implementation"""
    # Create market with single house
    market = HousingMarket([house])
    
    # Test house retrieval
    retrieved_house = market.get_house_by_id(1)
    assert retrieved_house == house, "Retrieved house should match original"
    
    # Test average price calculation
    avg_price = market.calculate_average_price(bedrooms=3)
    assert avg_price == 200000.0, "Incorrect average price calculation"
    
    # Test requirements filtering
    matching_houses = market.get_houses_that_meet_requirements(
        max_price=250000,
        segment=Segment.AVERAGE
    )
    assert isinstance(matching_houses, list), "Should return list of matching houses"
    assert len(matching_houses) == 1, "Should find one matching house"
    
    return market


def test_consumer_functionality(market: HousingMarket):
    """Test Consumer class implementation"""
    consumer = Consumer(
        id=1,
        annual_income=80000.0,
        children_number=2,
        segment=Segment.AVERAGE,
        house=None,
        savings=20000.0,
        saving_rate=0.3,
        interest_rate=0.05
    )
    
    # Test savings calculation
    initial_savings = consumer.savings
    consumer.compute_savings(years=5)
    assert consumer.savings > initial_savings, "Savings should increase over time"
    
    # Test house purchase
    consumer.buy_a_house(market)
    assert consumer.house is not None or market.get_houses_that_meet_requirements(
        max_price=consumer.savings * 5,  # Assuming 20% down payment
        segment=consumer.segment
    ) is None, "Consumer should either buy a house or no suitable houses available"
    
    return consumer

def test_simulation(cleaned_data: List[Dict[str, Any]]):
    """Test Simulation class implementation"""
    simulation = Simulation(
        housing_market_data=cleaned_data,
        consumers_number=100,
        years=5,
        annual_income=AnnualIncomeStatistics(
            minimum=30000.0,
            average=60000.0,
            standard_deviation=20000.0,
            maximum=150000.0
        ),
        children_range=ChildrenRange(
            minimum=0,
            maximum=5
        ),
        down_payment_percentage=0.2,
        saving_rate=0.3,
        interest_rate=0.05,
        order=CleaningMarketMechanism.RANDOM
    )
    
    # Test market creation
    simulation.create_housing_market()
    assert hasattr(simulation, 'housing_market'), "Housing market should be created"
    
    # Test consumer creation
    simulation.create_consumers()
    assert hasattr(simulation, 'consumers'), "Consumers should be created"
    assert len(simulation.consumers) == 100, "Should create specified number of consumers"
    
    # Test savings computation
    simulation.compute_consumers_savings()
    assert all(c.savings > 0 for c in simulation.consumers), "All consumers should have savings"
    
    # Test market cleaning
    simulation.clean_the_market()
    
    # Test final statistics
    owners_rate = simulation.compute_owners_population_rate()
    assert 0 <= owners_rate <= 1, "Owners population rate should be between 0 and 1"
    
    availability_rate = simulation.compute_houses_availability_rate()
    assert 0 <= availability_rate <= 1, "Houses availability rate should be between 0 and 1"


# test simulation


from src.real_estate_toolkit.agent_based_model.simulation import (
    Simulation,
    AnnualIncomeStatistics,
    ChildrenRange,
    CleaningMarketMechanism,
)
from src.real_estate_toolkit.agent_based_model.houses import QualityScore

def test_simulation():
    # Example housing market data
    housing_market_data = [
        {"id": 1, "price": 300000, "area": 1500, "bedrooms": 3, "year_built": 2020, "quality_score": QualityScore.EXCELLENT.value},
        {"id": 2, "price": 250000, "area": 1200, "bedrooms": 2, "year_built": 2018, "quality_score": QualityScore.GOOD.value},
        {"id": 3, "price": 400000, "area": 1800, "bedrooms": 4, "year_built": 2019, "quality_score": QualityScore.EXCELLENT.value},
    ]

    # Initialize simulation parameters
    simulation = Simulation(
        housing_market_data=housing_market_data,
        consumers_number=10,
        years=5,
        annual_income=AnnualIncomeStatistics(minimum=30000, average=75000, standard_deviation=15000, maximum=150000),
        children_range=ChildrenRange(minimum=0, maximum=3),
        cleaning_market_mechanism=CleaningMarketMechanism.INCOME_ORDER_DESCENDANT
    )

    # Step 1: Create Consumers
    simulation.create_consumers()
    assert len(simulation.consumers) == 10, "Consumers were not created correctly."

    # Step 2: Compute Consumers' Savings
    simulation.compute_consumers_savings()
    for consumer in simulation.consumers:
        assert consumer.savings > 0, "Consumer savings not computed correctly."

    # Step 3: Clean the Market
    simulation.clean_the_market()
    owners_rate = simulation.compute_owners_population_rate()
    houses_availability_rate = simulation.compute_houses_availability_rate()

    # Assert final metrics
    assert 0 <= owners_rate <= 100, "Owners population rate is out of range."
    assert 0 <= houses_availability_rate <= 100, "Houses availability rate is out of range."

    # Print results for verification
    print(f"Owners population rate: {owners_rate}%")
    print(f"Houses availability rate: {houses_availability_rate}%")

if __name__ == "__main__":
    test_simulation()



