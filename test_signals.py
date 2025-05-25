#!/usr/bin/env python3
"""
Test script to verify the get_signals method works correctly
"""

def test_get_signals():
    try:
        # Test imports first
        from trading_variables import RSI_OVERSOLD, RSI_OVERBOUGHT, SMA_WINDOW
        print(f'✓ Imports successful: RSI_OVERSOLD={RSI_OVERSOLD}, RSI_OVERBOUGHT={RSI_OVERBOUGHT}, SMA_WINDOW={SMA_WINDOW}')
        
        # Test the method step by step
        from indicators import TechnicalAnalysis
        print('✓ TechnicalAnalysis imported successfully')
        
        # Create simple test data with sufficient periods
        test_prices = [100, 101, 99, 102, 98, 103, 97, 104, 96, 105, 95, 106, 94, 107, 93, 108, 92, 109, 91, 110, 90, 111, 89, 112, 88, 113, 87, 114, 86, 115]
        print(f'✓ Test data created: {len(test_prices)} periods')
        
        ta = TechnicalAnalysis(test_prices)
        print('✓ TechnicalAnalysis instance created')
        
        # Test calculate_indicators first
        indicators = ta.calculate_indicators()
        print('✓ calculate_indicators completed')
        print(f"  Key indicators: RSI={indicators.get('rsi', 'N/A'):.2f}, MACD={indicators.get('macd', 'N/A'):.4f}")
        
        # Test get_signals
        signals, indicators = ta.get_signals()
        print(f'✓ get_signals completed: {len(signals)} signals generated')
        
        if signals:
            for i, signal in enumerate(signals):
                print(f'  Signal {i+1}: {signal}')
        else:
            print('  No signals generated - checking why...')
            
            # Debug: Check required indicators
            required_indicators = ['current_price', 'sma_20', 'rsi', 'macd', 'obv', 'adi', 'vwap']
            print('  Required indicators status:')
            for req_ind in required_indicators:
                value = indicators.get(req_ind, 'MISSING')
                status = 'OK' if value is not None and value != 'MISSING' else 'MISSING/NULL'
                print(f'    {req_ind}: {value} ({status})')
        
        return True
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_get_signals()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
