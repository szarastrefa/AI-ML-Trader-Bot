# Broker Integration Documentation

This directory contains detailed integration guides for each supported broker and trading platform. Each guide provides step-by-step instructions for obtaining API keys, configuring accounts, and troubleshooting common issues.

## Supported Brokers

### Forex/CFD Brokers

#### MetaTrader 5 Platforms
- [XM Global](./xm_global.md) - MetaTrader 5 integration
- [IC Markets](./ic_markets.md) - MetaTrader 5 integration
- [Admiral Markets](./admiral_markets.md) - MetaTrader 5 integration
- [XTB](./xtb.md) - MetaTrader 5 integration
- [RoboForex](./roboforex.md) - MetaTrader 5 integration
- [InstaForex](./instaforex.md) - MetaTrader 5 integration
- [FBS](./fbs.md) - MetaTrader 5 integration

#### Other Forex Platforms
- [Interactive Brokers](./interactive_brokers.md) - TWS API integration
- [cTrader](./ctrader.md) - FIX API integration
- [NinjaTrader](./ninjatrader.md) - Native API integration
- [Currenex](./currenex.md) - FIX API integration
- [Hotspot FX](./hotspot_fx.md) - FIX API integration

### Stock/Futures Brokers
- [Interactive Brokers](./interactive_brokers.md) - Comprehensive guide
- [Tastyworks](./tastyworks.md) - REST API integration
- [Lightspeed](./lightspeed.md) - Native API integration
- [TradeStation](./tradestation.md) - REST/WebSocket API
- [RJO'Brien](./rjo_brien.md) - FIX API integration

### Cryptocurrency Exchanges

#### Major Exchanges (CCXT Support)
- [Binance](./binance.md) - Spot and futures trading
- [Coinbase Pro](./coinbase_pro.md) - Professional trading
- [Kraken](./kraken.md) - Spot and margin trading
- [Bitstamp](./bitstamp.md) - European exchange
- [Bitfinex](./bitfinex.md) - Advanced trading features
- [Gemini](./gemini.md) - Regulated US exchange

#### Additional Exchanges
- [Huobi](./huobi.md) - Global cryptocurrency exchange
- [OKX](./okx.md) - Derivatives and spot trading
- [Bybit](./bybit.md) - Derivatives specialist
- [KuCoin](./kucoin.md) - Wide asset selection
- [Bittrex](./bittrex.md) - US-based exchange

### Limited/No API Brokers
- [Plus500](./plus500.md) - **Note: No public API available**
- [eToro](./etoro.md) - Social trading platform (limited API)
- [Revolut Trading](./revolut.md) - Limited API access
- [Trading 212](./trading212.md) - No official API

## Quick Start Guide

### 1. Choose Your Broker
Select a broker from the supported list above based on your trading requirements:
- **Forex/CFD**: MetaTrader 5 brokers offer the best integration
- **Stocks/Options**: Interactive Brokers provides comprehensive access
- **Cryptocurrency**: Binance or Coinbase Pro for liquid markets

### 2. Create Account
Follow the broker-specific guide to:
- Create a trading account (live or demo)
- Complete account verification if required
- Fund your account (for live trading)

### 3. Obtain API Access
Each broker has different requirements:
- **MetaTrader 5**: Install terminal and enable algorithmic trading
- **Interactive Brokers**: Enable API access in account settings
- **Crypto Exchanges**: Generate API key with appropriate permissions

### 4. Configure Bot
Add broker credentials to your configuration:
```yaml
# config/brokers.yml
brokers:
  my_mt5_account:
    type: mt5
    server: "MT5Server-Demo"
    login: 12345678
    password: "your_password"
    
  my_binance_account:
    type: ccxt
    exchange: binance
    api_key: "your_api_key"
    api_secret: "your_api_secret"
    sandbox: true  # Use testnet for testing
```

### 5. Test Connection
Use the built-in health check endpoint:
```bash
curl http://localhost:5000/api/brokers/health
```

## General Requirements

### For All Brokers
- Stable internet connection
- Valid trading account (demo or live)
- Appropriate account permissions for API access
- Sufficient account balance for trading

### Security Considerations
- **Never commit API credentials to version control**
- Use environment variables for sensitive data
- Enable IP whitelisting when available
- Use read-only API keys for testing
- Regularly rotate API keys
- Monitor API usage limits

### Network Requirements
- **Firewall**: Allow outbound HTTPS (port 443) traffic
- **DNS**: Ensure broker domains are resolvable
- **Latency**: Consider server location for HFT strategies
- **Backup Connection**: Have redundant internet connections

## Common Integration Patterns

### MetaTrader 5 Integration
1. Install MT5 terminal on the same machine as the bot
2. Login to your trading account
3. Enable "Allow algorithmic trading" in settings
4. The bot connects via MT5 Python library

### REST API Integration
1. Obtain API credentials from broker
2. Configure rate limits and retry logic
3. Implement proper error handling
4. Use webhooks for real-time updates when available

### FIX Protocol Integration
1. Obtain FIX gateway credentials
2. Configure FIX session parameters
3. Implement proper message handling
4. Handle reconnection and sequence number gaps

### WebSocket Integration
1. Subscribe to relevant market data feeds
2. Implement reconnection logic
3. Handle rate limiting and throttling
4. Process messages asynchronously

## Troubleshooting

### Common Issues

#### Authentication Failures
- Verify API credentials are correct
- Check if API access is enabled on the account
- Ensure IP address is whitelisted
- Verify permissions for required operations

#### Connection Issues
- Check network connectivity
- Verify firewall settings
- Test DNS resolution for broker domains
- Check for service outages

#### Rate Limiting
- Implement exponential backoff
- Respect broker rate limits
- Use bulk operations when available
- Consider upgrading to higher API tiers

#### Data Issues
- Verify symbol format matches broker requirements
- Check market hours for the instrument
- Ensure sufficient market data subscriptions
- Validate timestamp formats

### Getting Help

1. **Check broker documentation** - Most issues are covered in official docs
2. **Review bot logs** - Look for specific error messages
3. **Test with minimal examples** - Isolate the problem
4. **Contact broker support** - For account-specific issues
5. **Community forums** - Many brokers have active communities

## API Limits and Costs

### Typical Rate Limits
- **REST APIs**: 100-1000 requests per minute
- **WebSocket**: 5-100 connections per account
- **Market Data**: May require separate subscriptions

### Cost Considerations
- **Market Data Fees**: Some brokers charge for real-time data
- **API Access Fees**: Premium accounts may have higher limits
- **Transaction Costs**: Consider spreads, commissions, and fees
- **Infrastructure Costs**: VPS or cloud hosting for low latency

## Best Practices

### Development
- Always test with demo/sandbox accounts first
- Implement comprehensive logging
- Use configuration management
- Build robust error handling
- Monitor API usage metrics

### Production
- Use production API endpoints
- Implement health checks and monitoring
- Set up alerts for connection failures
- Have rollback procedures ready
- Maintain emergency contact information

### Risk Management
- Start with small position sizes
- Implement circuit breakers
- Monitor exposure across all accounts
- Have manual override capabilities
- Regular strategy performance reviews

---

**Next Steps:**
1. Choose your preferred broker(s)
2. Follow the specific integration guide
3. Test the connection thoroughly
4. Configure risk management parameters
5. Deploy with monitoring

**Support:** For technical issues, please check the specific broker documentation or contact the broker's API support team.