import { createStore, applyMiddleware, combineReducers } from 'redux';
import promiseMiddleware from 'redux-promise';

const createStoreWithMiddleware = applyMiddleware(promiseMiddleware)(createStore);

const reducers = combineReducers({

});

const store = createStoreWithMiddleware(reducers);

export default store;
