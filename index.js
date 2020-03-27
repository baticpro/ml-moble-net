import * as mobilenet from '@tensorflow-models/mobilenet';
import * as tfnode from '@tensorflow/tfjs-node';
import * as fs from 'fs';

const loadAndDecodeImage = path => {
  const buffer = fs.readFileSync(path);
  return tfnode.node.decodeImage(buffer);
};

const getPredictions = async path => {
  const image = loadAndDecodeImage(path);
  const model = await mobilenet.load();
  return await model.classify(image);
};

const main = async () => {
  const predictions = await getPredictions('testing-images/IMG_5313.JPG');
  console.log('results: ', predictions);
};

main();
