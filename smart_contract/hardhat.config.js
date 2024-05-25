require('@nomiclabs/hardhat-waffle');

module.exports = {
  solidity: '0.8.0',
  networks: {
    sepolia: {
      url: 'https://eth-sepolia.g.alchemy.com/v2/QlZ3JksbeUn6-T61H2vYrRxT7njSptTN',
      accounts: ['16ae74a26635dff0e8b94385d4fc319dccdb17ff3153734e3fe4b41f567cd7d8'],
    },
  },
};