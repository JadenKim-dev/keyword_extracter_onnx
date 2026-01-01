/**
 * Common English stopwords for keyword filtering
 *
 * Based on standard NLP stopword lists. These are common words that
 * typically don't carry significant meaning when extracted as keywords.
 */
export const ENGLISH_STOPWORDS = new Set([
  // Articles
  'a', 'an', 'the',

  // Prepositions
  'in', 'on', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
  'into', 'through', 'during', 'before', 'after', 'above', 'below',
  'to', 'from', 'up', 'down', 'of', 'off', 'over', 'under',

  // Conjunctions
  'and', 'or', 'but', 'nor', 'so', 'yet', 'if', 'unless', 'although',
  'because', 'while', 'where', 'when', 'why', 'how',

  // Pronouns
  'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their',
  'my', 'your', 'his', 'her', 'its', 'our', 'this', 'that', 'these', 'those',

  // Common verbs
  'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
  'shall', 'should', 'may', 'might', 'can', 'could', 'must',

  // Other common words
  'not', 'no', 'yes', 'too', 'very', 'just', 'now', 'than', 'then',
  'once', 'here', 'there', 'all', 'both', 'each', 'few', 'more',
  'most', 'other', 'some', 'such', 'only', 'own', 'same', 'as', 'what',
]);

/**
 * Check if a word is a stopword (case-insensitive)
 *
 * @param word - The word to check
 * @param customStopwords - Optional custom stopwords set to use instead of default
 * @returns True if the word is a stopword
 *
 * @example
 * ```typescript
 * isStopword('the');  // true
 * isStopword('machine');  // false
 * isStopword('THE');  // true (case-insensitive)
 * ```
 */
export function isStopword(word: string, customStopwords?: Set<string>): boolean {
  const stopwords = customStopwords || ENGLISH_STOPWORDS;
  return stopwords.has(word.toLowerCase());
}
