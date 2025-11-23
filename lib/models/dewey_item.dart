class DeweyItem implements Comparable<DeweyItem> {
  final bool isNumeric;
  final String prefix;
  final double number;
  final String cutter;
  final bool isValid;

  DeweyItem({
    required this.isNumeric,
    required this.prefix,
    required this.number,
    required this.cutter,
    this.isValid = true, // Par défaut true
  });

  @override
  int compareTo(DeweyItem other) {
    // Si l'un des deux est invalide, on ne devrait théoriquement pas comparer,
    // mais pour la stabilité du tri, on les renvoie à la fin ou au début.
    if (!isValid && other.isValid) return 1; 
    if (isValid && !other.isValid) return -1;
    
    // Le reste est inchangé...
    if (isNumeric && !other.isNumeric) return -1;
    if (!isNumeric && other.isNumeric) return 1;

    if (isNumeric) {
      int numComp = number.compareTo(other.number);
      if (numComp != 0) return numComp;
      return cutter.compareTo(other.cutter);
    } else {
      int preComp = prefix.compareTo(other.prefix);
      if (preComp != 0) return preComp;
      return cutter.compareTo(other.cutter);
    }
  }
}