import * as fs from 'fs';

const file = process.argv[2];

let content = fs.readFileSync(file, 'utf8');

content = content
    // .replace(/，/g, ', ')

fs.writeFileSync(file, content, 'utf8');

console.log(`Updated ${file} successfully!`);
