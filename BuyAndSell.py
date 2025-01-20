#key PKT46WSFY5344MCWJXU2
#secret  69yZ4Fvk29sTBW88A3nI9YgEvzfV4cE2fnZnXMnv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# paper=True enables paper trading
trading_client = TradingClient('PKT46WSFY5344MCWJXU2', '69yZ4Fvk29sTBW88A3nI9YgEvzfV4cE2fnZnXMnv', paper=True)

# preparing orders
market_order_data = MarketOrderRequest(
                    symbol="AAPL",
                    qty=0.023,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                    )

# Market order
market_order = trading_client.submit_order(
                order_data=market_order_data
               )