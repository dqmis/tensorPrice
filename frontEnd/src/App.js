import React from 'react';
import axios from 'axios';
import './App.css';

class App extends React.Component {
	constructor(props) {
		super(props)

		this.state = { selectedFile: null, loading: null, retailer: null, price: null, err: null };
		this.fileChangeHandler = this.fileChangeHandler.bind(this);
		this.uploadHandler = this.uploadHandler.bind(this);
	}

	fileChangeHandler = (event) => {
		this.setState({ selectedFile: event.target.files[0], retailer: null, price: null, err: null },
			() => this.uploadHandler());
	}

	uploadHandler = () => {
		this.setState.loading = true;
		const formData = new FormData();
		formData.append('file', this.state.selectedFile, this.state.selectedFile.name)

		const self = this;
		axios.post('http://localhost:8080', formData)
			.then(function(resp) {
				const retailer = resp.data.Shop;
				const price = resp.data.Price;
				self.setState({ retailer, price, selectedFile: null })
			})
			.catch(function(err) {
				self.setState({ selectedFile: null, err: 'Bad file or request :(' });
			});
	}

	render() {
		return (
			<div className="container">
				<div className="d-flex justify-content-center">
					<div className="p-8">
						<h2 className="title">tensorPrice</h2>
						<h3 className="subtitle">To start just upload a receipt.</h3>
					</div>
				</div>
				<div className="d-flex justify-content-center">
					<div className="p-8"> { this.state.selectedFile === null &&
						<label class="btn btn-default upload-button">
							Upload
							<input type="file" hidden onChange={ this.fileChangeHandler }/>
						</label> } { this.state.selectedFile !== null &&
							<div className="lds-dual-ring"></div> }
					</div>
				</div>
				{ this.state.retailer != null && 
				<div className="d-flex justify-content-center">
					<div className="p-8">
						<h2 className="result-retailer">{ this.state.retailer }</h2>
						<h2 className="result-price">{ this.state.price } €</h2>
					</div>
				</div> } { this.state.err !== null &&
						<h2 className="error">{ this.state.err }</h2>}
			</div>
		);
	}
}

export default App;
