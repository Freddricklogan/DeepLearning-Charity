/** Report page behaviour: mounts the Executive Shell from the embedded JSON. */
import { mountExecShell } from './exec-shell.js';

const data = JSON.parse(document.getElementById('report-data').textContent);
const pct = (v, d = 2) => `${(v * 100).toFixed(d)}%`;

const shell = mountExecShell({
  title: 'Charity Funding Outcome Classifier',
  tagline: `A tabular classifier on the Alphabet Soup dataset with the parts a reviewer looks for: tested preprocessing, a logistic-regression baseline beside the network, metrics measured on held-out rows in the CI run that published this page, and a generated model card that says what the model is not for.`,
  repo: 'https://github.com/Freddricklogan/DeepLearning-Charity',
  pagesUrl: 'https://freddricklogan.github.io/DeepLearning-Charity/',
  badges: [{ label: 'Keras 3 · JAX', tone: 'accent' }, { label: 'Baseline compared', dot: true }, { label: 'Measured in CI', dot: true }],
  kpis: [
    { label: 'Network accuracy', compute: () => pct(data.network.accuracy), tone: 'accent' },
    { label: 'Logistic baseline', compute: () => pct(data.baseline.accuracy) },
    { label: 'Network ROC AUC', compute: () => data.network.roc_auc.toFixed(3), tone: 'ok' },
    { label: 'Majority class', compute: () => pct(Math.max(data.baseRate, 1 - data.baseRate)), tone: 'muted' },
    { label: 'Epochs (best)', compute: () => `${data.epochsRun} (${data.bestEpoch})`, tone: 'warn' }
  ],
  tour: [
    { selector: '.cm-note', title: 'What this page is', body: `The CI pipeline ran the package on the vendored dataset with seed ${data.seed}: ${data.rows.toLocaleString()} rows, ${data.features} features after encoding, ${data.testRows.toLocaleString()} held-out rows. Every number here came from that run.` },
    { selector: '#s-results', title: 'The baseline is the point', body: `Logistic regression reaches ${pct(data.baseline.accuracy)}; the network reaches ${pct(data.network.accuracy)}. The gap is ${((data.network.accuracy - data.baseline.accuracy) * 100).toFixed(2)} points on one split — the card says to treat that as noise unless it repeats across seeds.` },
    { selector: '#s-training', title: 'Early stopping, shown', body: `Training and validation loss by epoch; the run stopped after ${data.epochsRun} epochs and restored the weights from epoch ${data.bestEpoch}.` },
    { selector: '#s-calibration', title: 'Probabilities, not just labels', body: 'The reliability table compares predicted probability with the observed rate in each bin, and the threshold sweep shows what moving the operating point does to precision and recall.' }
  ]
});
shell.refreshKpis();
