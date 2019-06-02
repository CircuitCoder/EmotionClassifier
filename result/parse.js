const fs = require('fs');

const log = fs.readFileSync('./log.txt').toString('utf-8').split('\n');

const rows = [];

for(const row of log) {
	if(row.match(/^Epoch: +\d+$/)) {
		const inner = row.match(/^Epoch: +(\d+)$/)[1]
		rows.push({
			epoch: parseInt(inner) + 1,
		});
	}

	if(row.match(/^Train: +[0-9.]+%$/)) {
		const inner = row.match(/^Train: +([0-9.]+)%$/)[1]
		rows[rows.length-1].train = parseFloat(inner);
	}

	if(row.match(/^Acc: +[0-9.]+%$/)) {
		const inner = row.match(/^Acc: +([0-9.]+)%$/)[1]
		rows[rows.length-1].acc = parseFloat(inner);
	}
}

let rowA = '';
let rowB = '';
for(const row of rows){
	rowA += `(${row.epoch},${row.train})`;
	rowB += `(${row.epoch},${row.acc})`;
}

console.log(rowA);
console.log(rowB);
