#!/usr/bin/env python3
"""
CCXT Connector for Cryptocurrency Exchanges
Supports: Binance, Coinbase Pro, Kraken, Bitfinex, Huobi, OKX, Bybit, KuCoin, Bittrex, etc.
"""

import ccxt
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from .base_connector import (
    BaseConnector, Position, Order, Balance, Ticker, OrderBook, 
    OrderBookEntry, HistoricalBar, OrderType, OrderSide, OrderStatus
)

logger = logging.getLogger(__name__)


class CCXTConnector(BaseConnector):
    """CCXT connector for cryptocurrency exchanges"""
    
    def __init__(self, broker_name: str = "binance"):
        super().__init__(broker_name)
        self.exchange = None
        self.exchange_class = None
        self.markets = {}
        self.symbols = []
        
    def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to cryptocurrency exchange
        
        Args:
            config: Dict with 'api_key', 'secret', 'sandbox' (optional), 'passphrase' (for some exchanges)
            
        Returns:
            bool: True if connected successfully
        """
        try:
            # Get exchange class
            exchange_name = self.broker_name.lower()
            if exchange_name == 'coinbase_pro':
                exchange_name = 'coinbasepro'
            elif exchange_name == 'okx':
                exchange_name = 'okx'
                
            if not hasattr(ccxt, exchange_name):
                self.set_last_error(f"Exchange {exchange_name} not supported by CCXT")
                return False
            
            self.exchange_class = getattr(ccxt, exchange_name)
            
            # Initialize exchange
            exchange_config = {
                'apiKey': config.get('api_key', ''),
                'secret': config.get('secret', ''),
                'timeout': 30000,
                'enableRateLimit': True,
            }
            
            # Add passphrase for exchanges that require it (like Coinbase Pro)
            if 'passphrase' in config:
                exchange_config['passphrase'] = config['passphrase']
            
            # Set sandbox mode if specified
            if config.get('sandbox', False):
                exchange_config['sandbox'] = True
            
            self.exchange = self.exchange_class(exchange_config)
            
            # Test connection by loading markets
            self.markets = self.exchange.load_markets()
            self.symbols = list(self.markets.keys())
            
            # Get account info to verify connection
            if config.get('api_key'):  # Only if API key provided
                balance = self.exchange.fetch_balance()
                self.account_info = {
                    'exchange': exchange_name,
                    'trading_fees': self.exchange.fees,
                    'markets_count': len(self.markets),
                    'has_api_key': bool(config.get('api_key'))
                }
            
            self.config = config
            self.is_connected = True
            
            logger.info(f"Connected to {exchange_name} exchange with {len(self.markets)} markets")
            return True
            
        except Exception as e:
            self.set_last_error(f"CCXT connection error: {str(e)}")
            logger.error(f"CCXT connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from exchange"""
        try:
            if self.exchange:
                # CCXT doesn't require explicit disconnection
                self.exchange = None
                self.is_connected = False
                logger.info(f"Disconnected from {self.broker_name}")
        except Exception as e:
            logger.error(f"Error disconnecting from {self.broker_name}: {e}")
    
    def get_balance(self) -> Balance:
        """Get account balance"""
        try:
            if not self.exchange:
                raise Exception("Not connected to exchange")
            
            balance_data = self.exchange.fetch_balance()
            
            # Calculate totals in base currency (usually USD or USDT)
            total = balance_data.get('total', {})
            free = balance_data.get('free', {})
            used = balance_data.get('used', {})
            
            # Try to get USD values, fallback to first available currency
            base_currencies = ['USD', 'USDT', 'USDC', 'EUR', 'BTC']
            currency = 'USD'
            total_value = 0.0
            free_value = 0.0
            used_value = 0.0
            
            for curr in base_currencies:
                if curr in total and total[curr] > 0:
                    currency = curr
                    total_value = total[curr]
                    free_value = free.get(curr, 0.0)
                    used_value = used.get(curr, 0.0)
                    break
            
            # If no base currency found, sum all balances (rough estimate)
            if total_value == 0.0:
                total_value = sum(total.values())
                free_value = sum(free.values())
                used_value = sum(used.values())
                currency = 'Multi'
            
            return Balance(
                total=total_value,
                available=free_value,
                used=used_value,
                currency=currency,
                equity=total_value  # For crypto, equity = total balance
            )
            
        except Exception as e:
            self.set_last_error(f"Error getting balance: {str(e)}")
            logger.error(f"Error getting CCXT balance: {e}")
            return Balance(0, 0, 0)
    
    def get_positions(self) -> List[Position]:
        """Get open positions (for margin/futures trading)"""
        try:
            if not self.exchange:
                raise Exception("Not connected to exchange")
            
            positions = []
            
            # Try to fetch positions if exchange supports it
            if self.exchange.has['fetchPositions']:
                position_data = self.exchange.fetch_positions()
                
                for pos in position_data:
                    if pos['size'] != 0:  # Only active positions
                        position = Position(
                            symbol=pos['symbol'],
                            side=pos['side'],
                            size=abs(pos['size']),
                            entry_price=pos['entryPrice'] or 0.0,
                            current_price=pos['markPrice'] or 0.0,
                            profit=pos['unrealizedPnl'] or 0.0,
                            profit_percentage=pos['percentage'] or 0.0,
                            timestamp=datetime.fromtimestamp(pos['timestamp'] / 1000) if pos['timestamp'] else datetime.now(),
                            broker_position_id=pos['id'] or '',
                            unrealized_pnl=pos['unrealizedPnl'] or 0.0
                        )
                        positions.append(position)
            else:
                # For spot trading, simulate positions from balance
                balance_data = self.exchange.fetch_balance()
                for currency, amount in balance_data['total'].items():
                    if amount > 0 and currency not in ['USD', 'USDT', 'USDC', 'EUR']:
                        try:
                            symbol = f"{currency}/USDT"
                            if symbol in self.symbols:
                                ticker = self.exchange.fetch_ticker(symbol)
                                positions.append(Position(
                                    symbol=symbol,
                                    side='long',
                                    size=amount,
                                    entry_price=ticker['last'],  # Approximate
                                    current_price=ticker['last'],
                                    profit=0.0,  # Would need historical data to calculate
                                    profit_percentage=0.0,
                                    timestamp=datetime.now(),
                                    broker_position_id=f"spot_{currency}"
                                ))
                        except:
                            continue  # Skip if can't get ticker
            
            return positions
            
        except Exception as e:
            self.set_last_error(f"Error getting positions: {str(e)}")
            logger.error(f"Error getting CCXT positions: {e}")
            return []
    
    def get_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders"""
        try:
            if not self.exchange:
                raise Exception("Not connected to exchange")
            
            if symbol:
                orders_data = self.exchange.fetch_open_orders(symbol)
            else:
                orders_data = self.exchange.fetch_open_orders()
            
            orders = []
            for order_data in orders_data:
                order = Order(
                    symbol=order_data['symbol'],
                    side=OrderSide.BUY if order_data['side'] == 'buy' else OrderSide.SELL,
                    order_type=self._ccxt_to_order_type(order_data['type']),
                    amount=order_data['amount'],
                    price=order_data['price'],
                    order_id=order_data['id'],
                    status=self._ccxt_to_order_status(order_data['status']),
                    filled_amount=order_data['filled'],
                    timestamp=datetime.fromtimestamp(order_data['timestamp'] / 1000) if order_data['timestamp'] else datetime.now(),
                    broker_order_id=order_data['id']
                )
                orders.append(order)
            
            return orders
            
        except Exception as e:
            self.set_last_error(f"Error getting orders: {str(e)}")
            logger.error(f"Error getting CCXT orders: {e}")
            return []
    
    def place_order(self, order: Order) -> str:
        """Place a new order"""
        try:
            if not self.exchange:
                raise Exception("Not connected to exchange")
            
            if not self.validate_order(order):
                raise Exception("Order validation failed")
            
            # Prepare order parameters
            order_type = order.order_type.value
            side = order.side.value
            amount = order.amount
            symbol = order.symbol
            price = order.price
            
            # Place order
            if order_type == 'market':
                result = self.exchange.create_market_order(symbol, side, amount)
            else:
                result = self.exchange.create_limit_order(symbol, side, amount, price)
            
            logger.info(f"Order placed successfully: {result['id']}")
            return result['id']
            
        except Exception as e:
            self.set_last_error(f"Error placing order: {str(e)}")
            logger.error(f"Error placing CCXT order: {e}")
            return ""
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        try:
            if not self.exchange:
                raise Exception("Not connected to exchange")
            
            # Need to find the symbol for the order (CCXT requirement)
            # Try to get order details first
            try:
                order_info = self.exchange.fetch_order(order_id)
                symbol = order_info['symbol']
            except:
                # If can't fetch order info, try common symbols
                symbol = None
                for test_symbol in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']:
                    if test_symbol in self.symbols:
                        symbol = test_symbol
                        break
            
            if symbol:
                result = self.exchange.cancel_order(order_id, symbol)
                logger.info(f"Order cancelled successfully: {order_id}")
                return True
            else:
                raise Exception("Could not determine symbol for order cancellation")
            
        except Exception as e:
            self.set_last_error(f"Error cancelling order: {str(e)}")
            logger.error(f"Error cancelling CCXT order: {e}")
            return False
    
    def modify_order(self, order_id: str, **kwargs) -> bool:
        """Modify an existing order (if supported)"""
        try:
            if not self.exchange:
                raise Exception("Not connected to exchange")
            
            if not self.exchange.has['editOrder']:
                raise Exception("Order modification not supported by this exchange")
            
            # Implementation would depend on exchange capabilities
            # Most exchanges require cancelling and creating new order
            return False
            
        except Exception as e:
            self.set_last_error(f"Error modifying order: {str(e)}")
            logger.error(f"Error modifying CCXT order: {e}")
            return False
    
    def get_ticker(self, symbol: str) -> Ticker:
        """Get current market ticker"""
        try:
            if not self.exchange:
                raise Exception("Not connected to exchange")
            
            ticker_data = self.exchange.fetch_ticker(symbol)
            
            return Ticker(
                symbol=symbol,
                bid=ticker_data.get('bid', 0),
                ask=ticker_data.get('ask', 0),
                last=ticker_data.get('last', 0),
                volume=ticker_data.get('baseVolume', 0),
                timestamp=datetime.fromtimestamp(ticker_data['timestamp'] / 1000) if ticker_data.get('timestamp') else datetime.now(),
                high=ticker_data.get('high', 0),
                low=ticker_data.get('low', 0),
                open=ticker_data.get('open', 0),
                close=ticker_data.get('close', 0),
                change=ticker_data.get('change', 0),
                change_percentage=ticker_data.get('percentage', 0)
            )
            
        except Exception as e:
            self.set_last_error(f"Error getting ticker: {str(e)}")
            logger.error(f"Error getting CCXT ticker: {e}")
            return Ticker(symbol, 0, 0, 0, 0, datetime.now())
    
    def get_order_book(self, symbol: str, limit: int = 10) -> OrderBook:
        """Get market depth"""
        try:
            if not self.exchange:
                raise Exception("Not connected to exchange")
            
            order_book_data = self.exchange.fetch_order_book(symbol, limit)
            
            bids = [OrderBookEntry(price=bid[0], amount=bid[1]) for bid in order_book_data['bids'][:limit]]
            asks = [OrderBookEntry(price=ask[0], amount=ask[1]) for ask in order_book_data['asks'][:limit]]
            
            return OrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.fromtimestamp(order_book_data['timestamp'] / 1000) if order_book_data.get('timestamp') else datetime.now()
            )
            
        except Exception as e:
            self.set_last_error(f"Error getting order book: {str(e)}")
            logger.error(f"Error getting CCXT order book: {e}")
            return OrderBook(symbol, [], [], datetime.now())
    
    def get_historical_data(self, symbol: str, timeframe: str, 
                          start_time: datetime, end_time: datetime) -> List[HistoricalBar]:
        """Get historical OHLCV data"""
        try:
            if not self.exchange:
                raise Exception("Not connected to exchange")
            
            # Convert datetime to milliseconds
            since = int(start_time.timestamp() * 1000)
            until = int(end_time.timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv_data = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            
            # Filter by end time
            ohlcv_data = [bar for bar in ohlcv_data if bar[0] <= until]
            
            bars = []
            for bar in ohlcv_data:
                historical_bar = HistoricalBar(
                    timestamp=datetime.fromtimestamp(bar[0] / 1000),
                    open=bar[1],
                    high=bar[2],
                    low=bar[3],
                    close=bar[4],
                    volume=bar[5] if len(bar) > 5 else 0
                )
                bars.append(historical_bar)
            
            return bars
            
        except Exception as e:
            self.set_last_error(f"Error getting historical data: {str(e)}")
            logger.error(f"Error getting CCXT historical data: {e}")
            return []
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        try:
            if not self.exchange:
                raise Exception("Not connected to exchange")
            
            return self.symbols
            
        except Exception as e:
            self.set_last_error(f"Error getting symbols: {str(e)}")
            logger.error(f"Error getting CCXT symbols: {e}")
            return []
    
    def update_market_data(self) -> None:
        """Update market data"""
        try:
            if self.exchange:
                # Refresh markets if needed
                self.markets = self.exchange.load_markets(reload=True)
        except Exception as e:
            logger.error(f"Error updating CCXT market data: {e}")
    
    def _ccxt_to_order_type(self, ccxt_type: str) -> OrderType:
        """Convert CCXT order type to OrderType enum"""
        type_map = {
            'market': OrderType.MARKET,
            'limit': OrderType.LIMIT,
            'stop': OrderType.STOP,
            'stop_limit': OrderType.STOP_LIMIT,
            'trailing_stop': OrderType.TRAILING_STOP
        }
        return type_map.get(ccxt_type, OrderType.MARKET)
    
    def _ccxt_to_order_status(self, ccxt_status: str) -> OrderStatus:
        """Convert CCXT order status to OrderStatus enum"""
        status_map = {
            'open': OrderStatus.PENDING,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'cancelled': OrderStatus.CANCELLED,
            'expired': OrderStatus.EXPIRED,
            'rejected': OrderStatus.REJECTED
        }
        return status_map.get(ccxt_status, OrderStatus.PENDING)
    
    def is_market_open(self, symbol: str) -> bool:
        """Check if market is open (crypto markets are usually 24/7)"""
        return True
    
    def get_minimum_order_size(self, symbol: str) -> float:
        """Get minimum order size for symbol"""
        try:
            if symbol in self.markets:
                limits = self.markets[symbol].get('limits', {})
                amount_limits = limits.get('amount', {})
                return amount_limits.get('min', 0.001)
            return 0.001
        except Exception:
            return 0.001
    
    def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for symbol"""
        try:
            if self.exchange and hasattr(self.exchange, 'fees'):
                trading_fees = self.exchange.fees.get('trading', {})
                return {
                    'maker': trading_fees.get('maker', 0.001),
                    'taker': trading_fees.get('taker', 0.001)
                }
            return {'maker': 0.001, 'taker': 0.001}
        except Exception:
            return {'maker': 0.001, 'taker': 0.001}