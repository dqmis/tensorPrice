module.exports = {
    "extends": "airbnb",
    "rules":{
        "indent": [2, "tab"],
        "react/jsx-indent": [2, "tab"],
        "no-tabs": 0,
		"linebreak-style": 0,
		"react/jsx-filename-extension": [1, { "extensions": [".js", ".jsx"] }],
    },
    "globals": {
        "window": true,
        "document": true
    },
};